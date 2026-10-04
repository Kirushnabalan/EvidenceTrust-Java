
def remove_java_comments_preserve_layout(code: str) -> str:
    if not isinstance(code, str):
        raise TypeError("code must be a string")

    NORMAL = 0
    LINE_COMMENT = 1
    BLOCK_COMMENT = 2
    STRING = 3
    CHAR = 4
    TEXT_BLOCK = 5

    state = NORMAL
    result = []

    i = 0
    n = len(code)

    while i < n:
        # NORMAL CODE

        if state == NORMAL:
            # Java text block: """
            if code.startswith('"""', i):
                result.extend(['"', '"', '"'])
                i += 3
                state = TEXT_BLOCK
                continue

            # Line comment
            if code.startswith("//", i):
                result.extend([" ", " "])
                i += 2
                state = LINE_COMMENT
                continue

            # Block comment
            if code.startswith("/*", i):
                result.extend([" ", " "])
                i += 2
                state = BLOCK_COMMENT
                continue

            ch = code[i]

            if ch == '"':
                result.append(ch)
                i += 1
                state = STRING
                continue

            if ch == "'":
                result.append(ch)
                i += 1
                state = CHAR
                continue

            result.append(ch)
            i += 1
            continue

        # LINE COMMENT

        if state == LINE_COMMENT:
            ch = code[i]

            if ch == "\n":
                result.append("\n")
                i += 1
                state = NORMAL
                continue

            if ch == "\r":
                result.append("\r")
                i += 1
                continue

            result.append(" ")
            i += 1
            continue

        # BLOCK COMMENT

        if state == BLOCK_COMMENT:
            if code.startswith("*/", i):
                result.extend([" ", " "])
                i += 2
                state = NORMAL
                continue

            ch = code[i]

            if ch == "\n":
                result.append("\n")

            elif ch == "\r":
                result.append("\r")

            else:
                result.append(" ")

            i += 1
            continue

        # STRING LITERAL

        if state == STRING:
            ch = code[i]

            result.append(ch)
            i += 1

            # Preserve escaped character.
            if ch == "\\" and i < n:
                result.append(code[i])
                i += 1
                continue

            if ch == '"':
                state = NORMAL

            continue

        # CHARACTER LITERAL

        if state == CHAR:
            ch = code[i]

            result.append(ch)
            i += 1

            # Preserve escaped character.
            if ch == "\\" and i < n:
                result.append(code[i])
                i += 1
                continue

            if ch == "'":
                state = NORMAL

            continue

        # JAVA TEXT BLOCK

        if state == TEXT_BLOCK:
            if code.startswith('"""', i):
                result.extend(['"', '"', '"'])
                i += 3
                state = NORMAL
                continue

            result.append(code[i])
            i += 1
            continue

    cleaned = "".join(result)

    # We deliberately preserve character count.
    if len(cleaned) != len(code):
        raise RuntimeError("Comment cleaning changed source length.")

    # We deliberately preserve line count.
    if cleaned.count("\n") != code.count("\n"):
        raise RuntimeError("Comment cleaning changed source line count.")

    return cleaned
