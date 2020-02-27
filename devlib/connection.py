class ConnectionBase:

    def __init__(self):
        self.target = None
        self._old_conn = None

    def __enter__(self):
        self._old_conn = self.target.set_connection(self)
        return self.target

    def __exit__(self, exc_type, exc_value, traceback):
        self.target.set_connection(self._old_conn)

