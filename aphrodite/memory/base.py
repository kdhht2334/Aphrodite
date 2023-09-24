class Memory:
    def __init__(self, role: str, name: str, content: str) -> None:
        self._role = role
        self._name = name
        self._content = content

    @property
    def role(self) -> str:
        return self._role

    @property
    def name(self) -> str:
        return self._name

    @property
    def content(self) -> str:
        return self._content
