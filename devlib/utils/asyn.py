#    Copyright 2013-2025 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


"""
Async-related utilities
"""

import abc
import asyncio
import contextvars
import functools
import itertools
import contextlib
import pathlib
import queue
import os.path
import inspect
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from weakref import WeakSet

from greenlet import greenlet
from typing import (Any, Callable, TypeVar, Type,
                    Optional, Coroutine, Tuple, Dict, cast, Set,
                    List, Union, AsyncContextManager,
                    Iterable, Awaitable)
from collections.abc import AsyncGenerator, Generator
from asyncio import Task, AbstractEventLoop
from inspect import Signature, BoundArguments
from contextvars import Context
from queue import SimpleQueue
from threading import local


def create_task(awaitable: Awaitable, name: Optional[str] = None) -> Task:
    """
    Create a new asyncio Task from an awaitable and set its name.

    :param awaitable: A coroutine or awaitable object to schedule.
    :param name: An optional name for the task. If None, attempts to use the awaitable's __qualname__.
    :returns: The created asyncio Task.
    """
    if isinstance(awaitable, asyncio.Task):
        task: Task = awaitable
    else:
        task = asyncio.create_task(cast(Coroutine, awaitable))
    if name is None:
        name = getattr(awaitable, '__qualname__', None)
    task.set_name(name)
    return task


def _close_loop(loop: Optional[AbstractEventLoop]) -> None:
    """
    Close an asyncio event loop after shutting down asynchronous generators and the default executor.

    :param loop: The event loop to close, or None.
    """
    if loop is not None:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
            try:
                shutdown_default_executor = loop.shutdown_default_executor
            except AttributeError:
                pass
            else:
                loop.run_until_complete(shutdown_default_executor())
        finally:
            loop.close()


class AsyncManager:
    """
    Manages asynchronous operations by tracking tasks and ensuring that concurrently
    running asynchronous functions do not interfere with one another.

    This manager maintains a mapping of tasks to resources and allows running tasks
    concurrently while checking for overlapping resource usage.
    """
    def __init__(self) -> None:
        """
        Initialize the AsyncManager with empty task trees and resource maps.
        """
        self.task_tree: Dict[Task, Set[Task]] = dict()
        self.resources: Dict[Task, Set['ConcurrentAccessBase']] = dict()

    def track_access(self, access: 'ConcurrentAccessBase') -> None:
        """
        Register the given ``access`` to have been handled by the current
        async task.

        :param access: Access that were done.

        This allows :func:`concurrently` to check that concurrent tasks did not
        step on each other's toes.
        """
        try:
            task: Optional[Task] = asyncio.current_task()
        except RuntimeError:
            pass
        else:
            if task:
                self.resources.setdefault(task, set()).add(access)

    async def concurrently(self, awaitables: Iterable[Awaitable]) -> List[Any]:
        """
        Await concurrently for the given awaitables, and cancel them as soon as
        one raises an exception.

        :param awaitables: An iterable of coroutine objects to run concurrently.
        :returns: A list with the results of the awaitables.
        :raises Exception: Propagates the first exception encountered, canceling the others.
        """
        awaitables_list: List[Awaitable] = list(awaitables)

        # Avoid creating asyncio.Tasks when it's not necessary, as it will
        # disable a the blocking path optimization of Target._execute_async()
        # that uses blocking calls as long as there is only one asyncio.Task
        # running on the event loop.
        if len(awaitables_list) == 1:
            return [await awaitables_list[0]]

        tasks: List[Task] = list(map(create_task, awaitables_list))

        current_task: Optional[Task] = asyncio.current_task()
        task_tree: Dict[Task, Set[Task]] = self.task_tree

        try:
            if current_task:
                node: Set[Task] = task_tree[current_task]
        except KeyError:
            is_root_task: bool = True
            node = set()
        else:
            is_root_task = False
        if current_task:
            task_tree[current_task] = node

        task_tree.update({
            child: set()
            for child in tasks
        })
        node.update(tasks)

        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            raise
        finally:

            def get_children(task: Task) -> frozenset[Task]:
                """
                get the children of the task and their children etc and return as a
                single set
                """
                immediate_children: Set[Task] = task_tree[task]
                return frozenset(
                    itertools.chain(
                        [task],
                        immediate_children,
                        itertools.chain.from_iterable(
                            map(get_children, immediate_children)
                        )
                    )
                )

            # Get the resources created during the execution of each subtask
            # (directly or indirectly)
            resources: Dict[Task, frozenset['ConcurrentAccessBase']] = {
                task: frozenset(
                    itertools.chain.from_iterable(
                        self.resources.get(child, [])
                        for child in get_children(task)
                    )
                )
                for task in tasks
            }
            for (task1, resources1), (task2, resources2) in itertools.combinations(resources.items(), 2):
                for res1, res2 in itertools.product(resources1, resources2):
                    if issubclass(res2.__class__, res1.__class__) and res1.overlap_with(res2):
                        raise RuntimeError(
                            'Overlapping resources manipulated in concurrent async tasks: {} (task {}) and {} (task {})'.format(res1, task1.get_name(), res2, task2.get_name())
                        )

            if is_root_task:
                self.resources.clear()
                task_tree.clear()

    async def map_concurrently(self, f: Callable, keys: Any) -> Dict:
        """
        Similar to :meth:`concurrently`,
        but maps the given function ``f`` on the given ``keys``.

        :param f: The function to apply to each key.
        :param keys: An iterable of keys.
        :return: A dictionary with ``keys`` as keys, and function result as
            values.
        """
        keys = list(keys)
        return dict(zip(
            keys,
            await self.concurrently(map(f, keys))
        ))


def compose(*coros: Callable) -> Callable[..., Coroutine]:
    """
    Compose coroutines, feeding the output of each as the input of the next
    one.

    ``await compose(f, g)(x)`` is equivalent to ``await f(await g(x))``

    :param coros: A variable number of coroutine functions.
    :returns: A callable that, when awaited, composes the coroutines in sequence.

    .. note:: In Haskell, ``compose f g h`` would be equivalent to ``f <=< g <=< h``
    """
    async def f(*args, **kwargs):
        empty_dict = {}
        for coro in reversed(coros):
            x = coro(*args, **kwargs)
            # Allow mixing corountines and regular functions
            if asyncio.isfuture(x):
                x = await x
            args = [x]
            kwargs = empty_dict

        return x
    return f


class _AsyncPolymorphicFunction:
    """
    A callable that allows exposing both a synchronous and asynchronous API.

    When called, the blocking synchronous operation is called. The ```asyn``
    attribute gives access to the asynchronous version of the function, and all
    the other attribute access will be redirected to the async function.

    :param asyn: The asynchronous version of the function.
    :param blocking: The synchronous (blocking) version of the function.
    """
    def __init__(self, asyn: Callable[..., Awaitable], blocking: Callable[..., Any]):
        self.asyn = asyn
        self.blocking = blocking
        functools.update_wrapper(self, asyn)

    def __get__(self, *args, **kwargs):
        return self.__class__(
            asyn=self.asyn.__get__(*args, **kwargs),
            blocking=self.blocking.__get__(*args, **kwargs),
        )

    # Ensure inspect.iscoroutinefunction() does not detect us as being async,
    # since __call__ is not.
    @property
    def __code__(self):
        return self.__call__.__code__

    def __call__(self, *args, **kwargs):
        return self.blocking(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.asyn, attr)


class memoized_method:
    """
    Decorator to memmoize a method.

    It works for:

        * async methods (coroutine functions)
        * non-async methods
        * method already decorated with :func:`devlib.asyn.asyncf`.

    :param f: The method to memoize.

    .. note:: This decorator does not rely on hacks to hash unhashable data. If
        such input is required, it will either have to be coerced to a hashable
        first (e.g. converting a list to a tuple), or the code of
        :func:`devlib.asyn.memoized_method` will have to be updated to do so.
    """
    def __init__(self, f: Callable):
        memo: 'memoized_method' = self

        sig: Signature = inspect.signature(f)

        def bind(self, *args: Any, **kwargs: Any) -> Tuple[Tuple[Any, ...],
                                                           Tuple[Any, ...],
                                                           Dict[str, Any]]:
            """
            bind arguments to function signature
            """
            bound: BoundArguments = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            key = (bound.args[1:], tuple(sorted(bound.kwargs.items())))

            return (key, bound.args, bound.kwargs)

        def get_cache(self) -> Dict[Tuple[Any, ...], Any]:
            try:
                cache: Dict[Tuple[Any, ...], Any] = self.__dict__[memo.name]
            except KeyError:
                cache = {}
                self.__dict__[memo.name] = cache
            return cache

        if inspect.iscoroutinefunction(f):
            @functools.wraps(f)
            async def async_wrapper(self, *args: Any, **kwargs: Any) -> Any:
                """
                wrapper for async functions
                """
                cache: Dict[Tuple[Any, ...], Any] = get_cache(self)
                key, args, kwargs = bind(self, *args, **kwargs)
                try:
                    return cache[key]
                except KeyError:
                    x = await f(*args, **kwargs)
                    cache[key] = x
                    return x
            self.f: Callable[..., Coroutine] = async_wrapper
        else:
            @functools.wraps(f)
            def sync_wrapper(self, *args: Any, **kwargs: Any) -> Any:
                """
                wrapper for sync functions
                """
                cache = get_cache(self)
                key, args, kwargs = bind(self, *args, **kwargs)
                try:
                    return cache[key]
                except KeyError:
                    x = f(*args, **kwargs)
                    cache[key] = x
                    return x
            self.f = sync_wrapper

        self._name = f.__name__

    @property
    def name(self) -> str:
        return '__memoization_cache_of_' + self._name

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __get__(self, obj: Optional['memoized_method'], owner: Optional[Type['memoized_method']] = None) -> Any:
        return self.f.__get__(obj, owner)

    def __set__(self, obj: 'memoized_method', value: Any):
        raise RuntimeError("Cannot monkey-patch a memoized function")

    def __set_name__(self, owner: Type['memoized_method'], name: str):
        self._name = name


class _Genlet(greenlet):
    """
    Generator-like object based on ``greenlets``. It allows nested :class:`_Genlet`
    to make their parent yield on their behalf, as if callees could decide to
    be annotated ``yield from`` without modifying the caller.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Forward the context variables to the greenlet, which will not happen
        # by default:
        # https://greenlet.readthedocs.io/en/latest/contextvars.html
        self.gr_context = contextvars.copy_context()

    @classmethod
    def from_coro(cls, coro: Coroutine) -> '_Genlet':
        """
        Create a :class:`_Genlet` from a given coroutine, treating it as a
        generator.

        :param coro: The coroutine to wrap.
        :returns: A _Genlet that wraps the coroutine.
        """
        def f(value: Any) -> Any:
            return self.consume_coro(coro, value)
        self = cls(f)
        return self

    def consume_coro(self, coro: Coroutine, value: Any) -> Any:
        """
        Send ``value`` to ``coro`` then consume the coroutine, passing all its
        yielded actions to the enclosing :class:`_Genlet`. This allows crossing
        blocking calls layers as if they were async calls with `await`.

        :param coro: The coroutine to consume.
        :param value: The initial value to send.
        :returns: The final value returned by the coroutine.
        :raises StopIteration: When the coroutine is exhausted.
        """
        excep: Optional[BaseException] = None
        while True:
            try:
                if excep is None:
                    future = coro.send(value)
                else:
                    future = coro.throw(excep)

            except StopIteration as e:
                return e.value
            else:
                parent: Optional[greenlet] = self.parent
                # Switch back to the consumer that returns the values via
                # send()
                try:
                    value = parent.switch(future) if parent else None
                except BaseException as e:
                    excep = e
                    value = None
                else:
                    excep = None

    @classmethod
    def get_enclosing(cls) -> Optional['_Genlet']:
        """
        Get the immediately enclosing :class:`_Genlet` in the callstack or
        ``None``.

        :returns: The nearest _Genlet instance in the chain, or None if not found.
        """
        g = greenlet.getcurrent()
        while not (isinstance(g, cls) or g is None):
            g = g.parent
        return g

    def _send_throw(self, value: Optional['_Genlet'], excep: Optional[BaseException]) -> Any:
        """
        helper function to do switch to another genlet or throw exception

        :param value: The value to send to the parent.
        :param excep: The exception to throw in the parent, or None.
        :returns: The result returned from the parent's switch.
        :raises StopIteration: If the parent completes.
        """
        self.parent = greenlet.getcurrent()

        # Switch back to the function yielding values
        if excep is None:
            result = self.switch(value)
        else:
            result = self.throw(excep)

        if self:
            return result
        else:
            raise StopIteration(result)

    def gen_send(self, x: Optional['_Genlet']) -> Any:
        """
        Similar to generators' ``send`` method.

        :param x: The value to send.
        :returns: The value received from the parent.
        """
        return self._send_throw(x, None)

    def gen_throw(self, x: Optional[BaseException]):
        """
        Similar to generators' ``throw`` method.

        :param x: The exception to throw.
        :returns: The value received from the parent after handling the exception.
        """
        return self._send_throw(None, x)


class _AwaitableGenlet:
    """
    Wraps a coroutine with a :class:`_Genlet` to allow it to be awaited using
    the normal 'await' syntax.

    :param coro: The coroutine to wrap.
    """

    @classmethod
    def wrap_coro(cls, coro: Coroutine) -> Coroutine:
        """
        Wrap a coroutine inside an _AwaitableGenlet so that it becomes awaitable.

        :param coro: The coroutine to wrap.
        :returns: An awaitable version of the coroutine.
        """
        async def coro_f() -> Any:
            # Make sure every new task will be instrumented since a task cannot
            # yield futures on behalf of another task. If that were to happen,
            # the task B trying to do a nested yield would switch back to task
            # A, asking to yield on its behalf. Since the event loop would be
            # currently handling task B, nothing would handle task A trying to
            # yield on behalf of B, leading to a deadlock.
            loop: AbstractEventLoop = asyncio.get_running_loop()
            _install_task_factory(loop)

            # Create a top-level _AwaitableGenlet that all nested runs will use
            # to yield their futures
            _coro: '_AwaitableGenlet' = cls(coro)

            return await _coro

        return coro_f()

    def __init__(self, coro: Coroutine):
        self._coro = coro

    def __await__(self) -> Generator:
        """
        Make the _AwaitableGenlet awaitable.

        :returns: A generator that yields from the wrapped coroutine.
        """
        coro: Coroutine = self._coro
        is_started: bool = inspect.iscoroutine(coro) and coro.cr_running

        def genf() -> Generator:
            """
            generator function
            """
            gen = _Genlet.from_coro(coro)
            value: Optional[_Genlet] = None
            excep: Optional[BaseException] = None

            # The coroutine is already started, so we need to dispatch the
            # value from the upcoming send() to the gen without running
            # gen first.
            if is_started:
                try:
                    value = yield
                except BaseException as e:
                    excep = e

            while True:
                try:
                    if excep is None:
                        future = gen.gen_send(value)
                    else:
                        future = gen.gen_throw(excep)
                except StopIteration as e:
                    return e.value
                finally:
                    _set_current_context(gen.gr_context)

                try:
                    value = yield future
                except BaseException as e:
                    excep = e
                    value = None
                else:
                    excep = None

        gen = genf()
        if is_started:
            # Start the generator so it waits at the first yield point
            cast(_Genlet, gen).gen_send(None)

        return gen


def _allow_nested_run(coro: Coroutine) -> Coroutine:
    """
    If the current callstack does not have an enclosing _Genlet, wrap the coroutine
    using _AwaitableGenlet; otherwise, return the coroutine unchanged.

    :param coro: The coroutine to potentially wrap.
    :returns: The original coroutine or a wrapped awaitable coroutine.
    """
    if _Genlet.get_enclosing() is None:
        return _AwaitableGenlet.wrap_coro(coro)
    else:
        return coro


def allow_nested_run(coro: Coroutine) -> Coroutine:
    """
    Wrap the coroutine ``coro`` such that nested calls to :func:`run` will be
    allowed. This is useful when a coroutine needs to yield control to another layer.

    .. warning:: The coroutine needs to be consumed in the same OS thread it
        was created in.

    :param coro: The coroutine to wrap.
    :returns: A possibly wrapped coroutine that allows nested execution.
    """
    return _allow_nested_run(coro)


# This thread runs coroutines that cannot be ran on the event loop in the
# current thread. Instead, they are scheduled in a separate thread where
# another event loop has been setup, so we can wrap coroutines before
# dispatching them there.
_CORO_THREAD_EXECUTOR = ThreadPoolExecutor(
    # Allow for a ridiculously large number so that we will never end up
    # queuing one job after another. This is critical as we could otherwise end
    # up in deadlock, if a job triggers another job and waits for it.
    max_workers=2**64,
)


def _check_executor_alive(executor: ThreadPoolExecutor) -> bool:
    """
    Check if the given ThreadPoolExecutor is still alive by submitting a no-op job.

    :param executor: The ThreadPoolExecutor to check.
    :returns: True if the executor accepts new jobs; False otherwise.
    """
    try:
        executor.submit(lambda: None)
    except RuntimeError:
        return False
    else:
        return True


_PATCHED_LOOP_LOCK = threading.Lock()
_PATCHED_LOOP: WeakSet = WeakSet()


def _install_task_factory(loop: AbstractEventLoop):
    """
    Install a task factory on the given event ``loop`` so that top-level
    coroutines are wrapped using :func:`allow_nested_run`. This ensures that
    the nested :func:`run` infrastructure will be available.

    :param loop: The asyncio event loop on which to install the task factory.
    """
    def install(loop: AbstractEventLoop) -> None:
        """
        install the task factory on the event loop
        """
        if sys.version_info >= (3, 11):
            def default_factory(loop: AbstractEventLoop, coro: Coroutine, context: Optional[Context] = None) -> Optional[Task]:
                return asyncio.Task(coro, loop=loop, context=context)
        else:
            def default_factory(loop: AbstractEventLoop, coro: Coroutine, context: Optional[Context] = None) -> Optional[Task]:
                return asyncio.Task(coro, loop=loop)

        make_task = loop.get_task_factory() or default_factory

        def factory(loop: AbstractEventLoop, coro: Coroutine, context: Optional[Context] = None) -> Optional[Task]:
            # Make sure each Task will be able to yield on behalf of its nested
            # await beneath blocking layers
            coro = _AwaitableGenlet.wrap_coro(coro)
            return cast(Callable, make_task)(loop, coro, context=context)

        loop.set_task_factory(cast(Callable, factory))

    with _PATCHED_LOOP_LOCK:
        if loop in _PATCHED_LOOP:
            return
        else:
            install(loop)
            _PATCHED_LOOP.add(loop)


def _set_current_context(ctx: Optional[Context]) -> None:
    """
    Get all the variable from the passed ``ctx`` and set them in the current
    context.

    :param ctx: A Context object containing variable values to set.
    """
    if ctx:
        for var, val in ctx.items():
            var.set(val)


class _CoroRunner(abc.ABC):
    """
    ABC for an object that can execute multiple coroutines in a given
    environment.

    This allows running coroutines for which it might be an assumption, such as
    the awaitables yielded by an async generator that are all attached to a
    single event loop.
    """
    @abc.abstractmethod
    def _run(self, coro: Coroutine) -> Any:
        """
        Execute the given coroutine using the runner's mechanism.

        :param coro: The coroutine to run.
        """
        pass

    def run(self, coro: Coroutine) -> Any:
        """
        Run the provided coroutine using the implemented runner. Raises an
        assertion error if the coroutine is already running.

        :param coro: The coroutine to run.
        :returns: The result of the coroutine.
        """
        # Ensure we have a fresh coroutine. inspect.getcoroutinestate() does not
        # work on all objects that asyncio creates on some version of Python, such
        # as iterable_coroutine
        assert not (inspect.iscoroutine(coro) and coro.cr_running)
        return self._run(coro)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        pass


class _ThreadCoroRunner(_CoroRunner):
    """
    Run the coroutines on a thread picked from a
    :class:`concurrent.futures.ThreadPoolExecutor`.

    Critically, this allows running multiple coroutines out of the same thread,
    which will be reserved until the runner ``__exit__`` method is called.

    :param future: A Future representing the thread running the coroutine loop.
    :param jobq: A SimpleQueue for scheduling coroutine jobs.
    :param resq: A SimpleQueue to collect results from executed coroutines.
    """
    def __init__(self, future: 'Future', jobq: 'SimpleQueue[Optional[Tuple[Context, Coroutine]]]',
                 resq: 'SimpleQueue[Tuple[Context, Optional[BaseException], Any]]'):
        self._future = future
        self._jobq = jobq
        self._resq = resq

    @staticmethod
    def _thread_f(jobq: 'SimpleQueue[Optional[Tuple[Context, Coroutine]]]',
                  resq: 'SimpleQueue[Tuple[Context, Optional[BaseException], Any]]') -> None:
        """
        Thread function that continuously processes scheduled coroutine jobs.

        :param jobq: Queue of jobs.
        :param resq: Queue to store results from the jobs.
        """
        def handle_jobs(runner: _LoopCoroRunner) -> None:
            while True:
                job: Optional[Tuple[Context, Coroutine]] = jobq.get()
                if job is None:
                    return
                else:
                    ctx, coro = job
                    try:
                        value: Any = ctx.run(runner.run, coro)
                    except BaseException as e:
                        value = None
                        excep: Optional[BaseException] = e
                    else:
                        excep = None

                    resq.put((ctx, excep, value))

        with _LoopCoroRunner(None) as runner:
            handle_jobs(runner)

    @classmethod
    def from_executor(cls, executor: ThreadPoolExecutor) -> '_ThreadCoroRunner':
        """
        Create a _ThreadCoroRunner by submitting the thread function to an executor.

        :param executor: A ThreadPoolExecutor to run the coroutine loop.
        :returns: An instance of _ThreadCoroRunner.
        :raises RuntimeError: If the executor is not alive.
        """
        jobq: SimpleQueue[Optional[Tuple[Context, Coroutine]]] = queue.SimpleQueue()
        resq: SimpleQueue = queue.SimpleQueue()

        try:
            future: Future = executor.submit(cls._thread_f, jobq, resq)
        except RuntimeError as e:
            if _check_executor_alive(executor):
                raise e
            else:
                raise RuntimeError('Devlib relies on nested asyncio implementation requiring threads. These threads are not available while shutting down the interpreter.')

        return cls(
            jobq=jobq,
            resq=resq,
            future=future,
        )

    def _run(self, coro: Coroutine) -> Any:
        """
        Schedule and run a coroutine in the separate thread, waiting for its result.

        :param coro: The coroutine to execute.
        :returns: The result from running the coroutine.
        :raises Exception: Propagates any exception raised by the coroutine.
        """
        ctx = contextvars.copy_context()
        self._jobq.put((ctx, coro))
        ctx, excep, value = self._resq.get()

        _set_current_context(ctx)

        if excep is None:
            return value
        else:
            raise excep

    def __exit__(self, *args, **kwargs):
        self._jobq.put(None)
        self._future.result()


class _LoopCoroRunner(_CoroRunner):
    """
    Run a coroutine on the given event loop.

    The passed event loop is assumed to not be running. If ``None`` is passed,
    a new event loop will be created in ``__enter__`` and closed in
    ``__exit__``.

    :param loop: An event loop to use; if None, a new one is created.
    """
    def __init__(self, loop: Optional[AbstractEventLoop]):
        self.loop = loop
        self._owned: bool = False

    def _run(self, coro: Coroutine) -> Any:
        """
        Run the given coroutine to completion on the event loop and return its result.

        :param coro: The coroutine to run.
        :returns: The result of the coroutine.
        """
        loop = self.loop

        # Back-propagate the contextvars that could have been modified by the
        # coroutine. This could be handled by asyncio.Runner().run(...,
        # context=...) or loop.create_task(..., context=...) but these APIs are
        # only available since Python 3.11
        ctx: Optional[Context] = None

        async def capture_ctx() -> Any:
            nonlocal ctx
            try:
                return await _allow_nested_run(coro)
            finally:
                ctx = contextvars.copy_context()

        try:
            if loop:
                return loop.run_until_complete(capture_ctx())
        finally:
            _set_current_context(ctx)

    def __enter__(self) -> '_LoopCoroRunner':
        loop: Optional[AbstractEventLoop] = self.loop
        if loop is None:
            owned = True
            loop = asyncio.new_event_loop()
        else:
            owned = False

        asyncio.set_event_loop(loop)

        self.loop = loop
        self._owned = owned
        return self

    def __exit__(self, *args, **kwargs):
        if self._owned:
            asyncio.set_event_loop(None)
            _close_loop(self.loop)


class _GenletCoroRunner(_CoroRunner):
    """
    Run a coroutine assuming one of the parent coroutines was wrapped with
    :func:`allow_nested_run`.

    :param g: The enclosing _Genlet instance.
    """
    def __init__(self, g: _Genlet):
        self._g = g

    def _run(self, coro: Coroutine) -> Any:
        """
        Execute the coroutine by delegating to the enclosing _Genlet's consume_coro method.

        :param coro: The coroutine to run.
        :returns: The result of the coroutine.
        """
        return self._g.consume_coro(coro, None)


def _get_runner() -> Union[_GenletCoroRunner,
                           _LoopCoroRunner,
                           _ThreadCoroRunner]:
    """
    Determine the appropriate coroutine runner based on the current context.
    Returns a _GenletCoroRunner if an enclosing _Genlet is present, a _LoopCoroRunner
    if an event loop exists (or can be created), or a _ThreadCoroRunner if an event loop is running.

    :returns: A coroutine runner appropriate for the current execution context.
    """
    executor: ThreadPoolExecutor = _CORO_THREAD_EXECUTOR
    g = _Genlet.get_enclosing()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    # We have an coroutine wrapped with allow_nested_run() higher in the
    # callstack, that we will be able to use as a conduit to yield the
    # futures.
    if g is not None:
        return _GenletCoroRunner(g)
    # No event loop setup, so we can just make our own
    elif loop is None:
        return _LoopCoroRunner(None)
    # There is an event loop setup, but it is not currently running so we
    # can just re-use it.
    #
    # TODO: for now, this path is dead since asyncio.get_running_loop() will
    # always raise a RuntimeError if the loop is not running, even if
    # asyncio.set_event_loop() was used.
    elif not loop.is_running():
        return _LoopCoroRunner(loop)
    # There is an event loop currently running in our thread, so we cannot
    # just create another event loop and install it since asyncio forbids
    # that. The only choice is doing this in a separate thread that we
    # fully control.
    else:
        return _ThreadCoroRunner.from_executor(executor)


def run(coro: Coroutine) -> Any:
    """
    Similar to :func:`asyncio.run` but can be called while an event loop is
    running if a coroutine higher in the callstack has been wrapped using
    :func:`allow_nested_run`.

    Note that context variables from :mod:`contextvars` will be available in
    the coroutine, and unlike with :func:`asyncio.run`, any update to them will
    be reflected in the context of the caller. This allows context variable
    updates to cross an arbitrary number of run layers, as if all those layers
    were just part of the same coroutine.

    :param coro: The coroutine to execute.
    :returns: The result of the coroutine.
    """
    runner = _get_runner()
    with runner as runner:
        return runner.run(coro)


def asyncf(f: Callable):
    """
    Decorator used to turn a coroutine into a blocking function, with an
    optional asynchronous API.

    **Example**::

        @asyncf
        async def foo(x):
            await do_some_async_things(x)
            return x

        # Blocking call, just as if the function was synchronous, except it may
        # use asynchronous code inside, e.g. to do concurrent operations.
        foo(42)

        # Asynchronous API, foo.asyn being a corountine
        await foo.asyn(42)

    This allows the same implementation to be both used as blocking for ease of
    use and backward compatibility, or exposed as a corountine for callers that
    can deal with awaitables.

    :param f: The asynchronous function to decorate.
    :returns: A callable that runs f synchronously, with an asynchronous version available as .asyn.
    """
    @functools.wraps(f)
    def blocking(*args, **kwargs) -> Any:
        # Since run() needs a corountine, make sure we provide one
        async def wrapper() -> Generator:
            x = f(*args, **kwargs)
            # Async generators have to be consumed and accumulated in a list
            # before crossing a blocking boundary.
            if inspect.isasyncgen(x):

                def genf() -> Generator:
                    asyncgen = x.__aiter__()
                    while True:
                        try:
                            yield run(asyncgen.__anext__())
                        except StopAsyncIteration:
                            return

                return genf()
            else:
                return await x
        return run(wrapper())

    return _AsyncPolymorphicFunction(
        asyn=f,
        blocking=blocking,
    )


class _AsyncPolymorphicCMState:
    def __init__(self) -> None:
        self.nesting: int = 0
        self.runner: Optional[Union[_GenletCoroRunner,
                                    _LoopCoroRunner,
                                    _ThreadCoroRunner]] = None

    def _update_nesting(self, n: int) -> bool:
        x = self.nesting
        assert x >= 0
        x = x + n
        self.nesting = x
        return bool(x)

    def _get_runner(self) -> Optional[Union[_GenletCoroRunner,
                                      _LoopCoroRunner,
                                      _ThreadCoroRunner]]:
        runner = self.runner
        if runner is None:
            assert not self.nesting
            runner = _get_runner()
            runner.__enter__()
        self.runner = runner
        return runner

    def _cleanup_runner(self, force: bool = False) -> None:
        def cleanup() -> None:
            self.runner = None
            if runner is not None:
                runner.__exit__(None, None, None)

        runner = self.runner
        if force:
            cleanup()
        else:
            assert runner is not None
            if not self._update_nesting(0):
                cleanup()


class _AsyncPolymorphicCM:
    """
    Wrap an async context manager such that it exposes a synchronous API as
    well for backward compatibility.

    :param async_cm: The asynchronous context manager to wrap.
    """

    def __init__(self, async_cm: AsyncContextManager):
        self.cm = async_cm
        self._state: local = threading.local()

    def _get_state(self):
        """
        Retrieve or initialize the thread-local state for this context manager.

        :returns: The state object.
        :rtype: _AsyncPolymorphicCMState
        """
        try:
            return self._state.x
        except AttributeError:
            state = _AsyncPolymorphicCMState()
            self._state.x = state
            return state

    def _delete_state(self) -> None:
        """
        Delete the thread-local state.
        """
        try:
            del self._state.x
        except AttributeError:
            pass

    def __aenter__(self, *args, **kwargs):
        return self.cm.__aenter__(*args, **kwargs)

    def __aexit__(self, *args, **kwargs):
        return self.cm.__aexit__(*args, **kwargs)

    @staticmethod
    def _exit(state: _AsyncPolymorphicCMState) -> None:
        state._update_nesting(-1)
        state._cleanup_runner()

    def __enter__(self, *args, **kwargs) -> Any:
        state: _AsyncPolymorphicCMState = self._get_state()
        runner: Optional[Union[_GenletCoroRunner,
                               _LoopCoroRunner,
                               _ThreadCoroRunner]] = state._get_runner()

        # Increase the nesting count _before_ we start running the
        # coroutine, in case it is a recursive context manager
        state._update_nesting(1)

        try:
            coro: Coroutine = self.cm.__aenter__(*args, **kwargs)
            if runner:
                return runner.run(coro)
        except BaseException:
            self._exit(state)
            raise

    def __exit__(self, *args, **kwargs) -> Any:
        coro: Coroutine = self.cm.__aexit__(*args, **kwargs)

        state: _AsyncPolymorphicCMState = self._get_state()
        runner: Optional[Union[_GenletCoroRunner,
                               _LoopCoroRunner,
                               _ThreadCoroRunner]] = state._get_runner()

        try:
            if runner:
                return runner.run(coro)
        finally:
            self._exit(state)

    def __del__(self):
        self._get_state()._cleanup_runner(force=True)


T = TypeVar('T')


def asynccontextmanager(f: Callable[..., AsyncGenerator[T, None]]) -> Callable[..., _AsyncPolymorphicCM]:
    """
    Same as :func:`contextlib.asynccontextmanager` except that it can also be
    used with a regular ``with`` statement for backward compatibility.

    :param f: A callable that returns an asynchronous generator.
    :returns: A context manager supporting both synchronous and asynchronous usage.
    """
    f_int = contextlib.asynccontextmanager(f)

    @functools.wraps(f_int)
    def wrapper(*args: Any, **kwargs: Any) -> _AsyncPolymorphicCM:
        cm = f_int(*args, **kwargs)
        return _AsyncPolymorphicCM(cm)

    return wrapper


class ConcurrentAccessBase(abc.ABC):
    """
    Abstract Base Class for resources tracked by :func:`concurrently`.
    Subclasses must implement the method to determine if two resources overlap.
    """
    @abc.abstractmethod
    def overlap_with(self, other: 'ConcurrentAccessBase') -> bool:
        """
        Return ``True`` if the resource overlaps with the given one.

        :param other: Resources that should not overlap with ``self``.
        :returns: True if the two resources overlap; False otherwise.

        .. note:: It is guaranteed that ``other`` will be a subclass of our
            class.
        """


class PathAccess(ConcurrentAccessBase):
    """
    Concurrent resource representing a file access.

    :param namespace: Identifier of the namespace of the path. One of "target" or "host".

    :param path: Normalized path to the file.

    :param mode: Opening mode of the file. Can be ``"r"`` for read and ``"w"``
        for writing.
    """
    def __init__(self, namespace: str, path: str, mode: str):
        assert namespace in ('host', 'target')
        self.namespace = namespace
        assert mode in ('r', 'w')
        self.mode = mode
        self.path = os.path.abspath(path) if namespace == 'host' else os.path.normpath(path)

    def overlap_with(self, other: ConcurrentAccessBase) -> bool:
        """
        Check if this path access overlaps with another access, considering
        namespace, mode, and filesystem hierarchy.

        :param other: Another resource access instance.
        :returns: True if the two paths overlap (and one of the accesses is for writing), else False.
        """
        other_internal = cast('PathAccess', other)
        path1 = pathlib.Path(self.path).resolve()
        path2 = pathlib.Path(other_internal.path).resolve()
        return (
            self.namespace == other_internal.namespace and
            'w' in (self.mode, other_internal.mode) and
            (
                path1 == path2 or
                path1 in path2.parents or
                path2 in path1.parents
            )
        )

    def __str__(self):
        """
        Return a string representation of the PathAccess, including the path and mode.

        :returns: A string describing the path access.
        """
        mode = {
            'r': 'read',
            'w': 'write',
        }[self.mode]
        return '{} ({})'.format(self.path, mode)
