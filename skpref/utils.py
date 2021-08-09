class UnderDevError(Exception):
    """
    Error that we give when something is being tried that is planned to be
    developed in the package, but isn't yet done
    """

    def __init__(self, message):
        self.message = message
