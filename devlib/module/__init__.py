#    Copyright 2014-2025 ARM Limited
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
from inspect import isclass

from devlib.exception import TargetStableError
from devlib.utils.types import identifier
from devlib.utils.misc import walk_modules, get_logger
from typing import (Optional, Dict, Union, Type,
                    TYPE_CHECKING, Any)
if TYPE_CHECKING:
    from devlib.target import Target
_module_registry: Dict[str, Type['Module']] = {}


def register_module(mod: Type['Module']) -> None:
    """
    Modules are specified on :class:`~devlib.target.Target` or
    :class:`~devlib.platform.Platform` creation by name. In order to find the class
    associated with the name, the module needs to be registered with ``devlib``.
    This is accomplished by passing the module class into :func:`register_module`
    method once it is defined.

    .. note:: If you're wiring a module to be included as part of ``devlib`` code
            base, you can place the file with the module class under
            ``devlib/modules/`` in the source and it will be automatically
            enumerated. There is no need to explicitly register it in that case.

    The code snippet below illustrates an implementation of a hard reset function
    for an "Acme" device.

    .. code:: python

        import os
        from devlib import HardResetModule, register_module


        class AcmeHardReset(HardResetModule):

            name = 'acme_hard_reset'

            def __call__(self):
                # Assuming Acme board comes with a "reset-acme-board" utility
                os.system('reset-acme-board {}'.format(self.target.name))

        register_module(AcmeHardReset)
    """
    if not issubclass(mod, Module):
        raise ValueError('A module must subclass devlib.Module')

    if mod.name is None:
        raise ValueError('A module must define a name')

    try:
        existing = _module_registry[mod.name]
    except KeyError:
        pass
    else:
        if existing is not mod:
            raise ValueError(f'Module "{mod.name}" already exists')
    _module_registry[mod.name] = mod


class Module:
    """
    Modules add additional functionality to the core :class:`~devlib.target.Target`
    interface. Usually, it is support for specific subsystems on the target. Modules
    are instantiated as attributes of the :class:`~devlib.target.Target` instance.

    Modules implement discrete, optional pieces of functionality ("optional" in the
    sense that the functionality may or may not be present on the target device, or
    that it may or may not be necessary for a particular application).

    Every module (ultimately) derives from :class:`devlib.module.Module` class.  A
    module must define the following class attributes:

    :name: A unique name for the module. This cannot clash with any of the existing
        names and must be a valid Python identifier, but is otherwise free-form.
    :kind: This identifies the type of functionality a module implements, which in
        turn determines the interface implemented by the module (all modules of
        the same kind must expose a consistent interface). This must be a valid
        Python identifier, but is otherwise free-form, though, where possible,
        one should try to stick to an already-defined kind/interface, lest we end
        up with a bunch of modules implementing similar functionality but
        exposing slightly different interfaces.

        .. note:: It is possible to omit ``kind`` when defining a module, in
                    which case the module's ``name`` will be treated as its
                    ``kind`` as well.

    :stage: This defines when the module will be installed into a
            :class:`~devlib.target.Target`. Currently, the following values are
            allowed:

            :connected: The module is installed after a connection to the target has
                        been established. This is the default.
            :early: The module will be installed when a
                    :class:`~devlib.target.Target` is first created. This should be
                    used for modules that do not rely on a live connection to the
                    target.
            :setup: The module will be installed after initial setup of the device
                    has been performed. This allows the module to utilize assets
                    deployed during the setup stage for example 'Busybox'.

    Additionally, a module must implement a static (or class) method :func:`probe`:
    """
    name: Optional[str] = None
    kind: Optional[str] = None
    attr_name: Optional[str] = None
    stage: str = 'connected'

    @staticmethod
    def probe(target: 'Target') -> bool:
        """
        This method takes a :class:`~devlib.target.Target` instance and returns
        ``True`` if this module is supported by that target, or ``False`` otherwise.

        .. note:: If the module ``stage`` is ``"early"``, this method cannot assume
                that a connection has been established (i.e. it can only access
                attributes of the Target that do not rely on a connection).
        """
        raise NotImplementedError()

    @classmethod
    def install(cls, target: 'Target', **params: Type['Module']):
        """
        The default installation method will create an instance of a module (the
        :class:`~devlib.target.Target` instance being the sole argument) and assign it
        to the target instance attribute named after the module's ``kind`` (or
        ``name`` if ``kind`` is ``None``).

        It is possible to change the installation procedure for a module by overriding
        the default :func:`install` method. The method must have the following
        signature:

        .. method:: Module.install(cls, target, **kwargs)

            Install the module into the target instance.
        """
        attr_name: Optional[str] = cls.attr_name
        installed: Dict[str, 'Module'] = target._installed_modules

        try:
            if attr_name:
                mod: 'Module' = installed[attr_name]
        except KeyError:
            mod = cls(target, **params)
            mod.logger.debug(f'Installing module {cls.name}')

            if mod.probe(target):
                for name in (
                    attr_name,
                    identifier(cls.name),
                    identifier(cls.kind) if cls.kind else None,
                ):
                    if name is not None:
                        installed[name] = mod
                if cls.name:
                    target._modules[cls.name] = params
                return mod
            else:
                raise TargetStableError(f'Module "{cls.name}" is not supported by the target')
        else:
            raise ValueError(
                f'Attempting to install module "{cls.name}" but a module is already installed as attribute "{attr_name}": {mod}'
            )

    def __init__(self, target: 'Target'):
        self.target = target
        self.logger = get_logger(self.name or '')

    def __init_subclass__(cls, *args, **kwargs) -> None:
        super().__init_subclass__(*args, **kwargs)

        attr_name: Optional[str] = cls.kind or cls.name
        cls.attr_name = identifier(attr_name) if attr_name else None

        if cls.name is not None:
            register_module(cls)


class HardRestModule(Module):
    """
    .. attribute:: HardResetModule.kind

    "hard_reset"
    """

    kind: str = 'hard_reset'

    def __call__(self):
        """
        .. method:: HardResetModule.__call__()

        Must be implemented by derived classes.

        Implements hard reset for a target devices. The equivalent of physically
        power cycling the device.  This may be used by client code in situations
        where the target becomes unresponsive and/or a regular reboot is not
        possible.
        """
        raise NotImplementedError()


class BootModule(Module):
    """
    .. attribute:: BootModule.kind

    "boot"
    """

    kind: str = 'boot'

    def __call__(self):
        """
        .. method:: BootModule.__call__()

        Must be implemented by derived classes.

        Implements a boot procedure. This takes the device from (hard or soft)
        reset to a booted state where the device is ready to accept connections. For
        a lot of commercial devices the process is entirely automatic, however some
        devices (e.g. development boards), my require additional steps, such as
        interactions with the bootloader, in order to boot into the OS.
        """
        raise NotImplementedError()

    def update(self, **kwargs) -> None:
        """
        .. method:: Bootmodule.update(**kwargs)

        Update the boot settings. Some boot sequences allow specifying settings
        that will be utilized during boot (e.g. linux kernel boot command line). The
        default implementation will set each setting in ``kwargs`` as an attribute of
        the boot module (or update the existing attribute).
        """
        for name, value in kwargs.items():
            if not hasattr(self, name):
                raise ValueError('Unknown parameter "{}" for {}'.format(name, self.name))
            self.logger.debug('Updating "{}" to "{}"'.format(name, value))
            setattr(self, name, value)


class FlashModule(Module):
    """
    A Devlib module used for performing firmware or image flashing operations on a target device.

    This module provides an abstraction for managing device flashing, such as flashing new
    bootloaders, system images, or recovery partitions, depending on the target platform.

    The `kind` attribute identifies the type of this module and is used by Devlib's internal
    module management system to categorize and invoke the appropriate functionality.

    Attributes:
        kind (str): The unique identifier for this module type. For `FlashModule`, this is "flash".

    Typical Usage:
        This module is automatically loaded onto targets that support flashing operations,
        such as development boards or phones with bootloader access.

    Example:
        >>> if FlashModule.probe(target):
        >>>     flash = FlashModule(target)
        >>>     flash.install()
        >>>     flash.flash_image("/path/to/image.img", partition="boot")

    Note:
        Subclasses of FlashModule should implement the actual flashing logic, as this base
        class only provides the interface and identification mechanism.
    """
    kind: str = 'flash'

    def __call__(self, image_bundle: Optional[str] = None,
                 images: Optional[Dict[str, str]] = None,
                 boot_config: Any = None, connect: bool = True) -> None:
        """
        .. method:: __call__(image_bundle=None, images=None, boot_config=None, connect=True)

        Must be implemented by derived classes.

        Flash the target platform with the specified images.

        :param image_bundle: A compressed bundle of image files with any associated
                            metadata. The format of the bundle is specific to a
                            particular implementation.
        :param images: A dict mapping image names/identifiers to the path on the
                    host file system of the corresponding image file. If both
                    this and ``image_bundle`` are specified, individual images
                    will override those in the bundle.
        :param boot_config: Some platforms require specifying boot arguments at the
                            time of flashing the images, rather than during each
                            reboot. For other platforms, this will be ignored.
        :connect: Specifiy whether to try and connect to the target after flashing.
        """
        raise NotImplementedError()


def get_module(mod: Union[str, Type[Module]]) -> Type[Module]:
    def from_registry(mod: str):
        try:
            return _module_registry[mod]
        except KeyError:
            raise ValueError('Module "{}" does not exist'.format(mod))

    if isinstance(mod, str):
        try:
            return from_registry(mod)
        except ValueError:
            # If the lookup failed, we may have simply not imported Modules
            # from the devlib.module package. The former module loading
            # implementation was also pre-importing modules, so we need to
            # replicate that behavior since users are currently not expected to
            # have imported the module prior to trying to use it.
            walk_modules('devlib.module')
            return from_registry(mod)

    elif issubclass(mod, Module):
        return mod
    else:
        raise ValueError('Not a valid module: {}'.format(mod))
