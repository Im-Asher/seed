from enum import Enum

class LanguageEnum(Enum):
    CPP="C/C++"
    JAVA="Java"
    PYTHON="Python"
    GO="Go"
    JS="Js"
    PHP="PHP"
    CSHARP="C#"
    RUST="Rust"
    RUBY="Ruby"
    NET=".NET"

class ResponseCode(Enum):
    Fail = 0
    Success = 1
    Exception = 4