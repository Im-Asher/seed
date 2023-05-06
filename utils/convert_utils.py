import re


class VersionConvert:
    __version_pattern = r'\b\d+\.\d+(?:\.\d+)*(?:\w+|-\w+)?|\d+\b'

    def convert(self, entity: str, label: str):
        entity = entity.lower()

        entity = self.__preprocess(entity)

        if label == "VERL":
            return self.__convert_to_version_list(entity=entity)
        if label == "VERR":
            return self.__convert_to_version_range(entity=entity)

    def __convert_to_version_list(self, entity: str):
        version_intervals = [(match.group(), match.start(), match.end())
                             for match in re.finditer(self.__version_pattern, entity)]

        versions = [v[0] for v in version_intervals]

        versions = ','.join(versions)

        return versions

    def __convert_to_version_range(self, entity: str):
        one_left = ['start', 'from', '>', '>=']
        one_right = ['prior', 'before', 'through',
                     'to', 'up', 'earlier', '<', '<=', 'below']

        versions = [match.group()
                    for match in re.finditer(self.__version_pattern, entity)]

        versions = sorted(versions)

        if len(versions) < 1:
            s = entity.find('all')
            if s != -1:
                return f'(,)'
            return f'()'

        if len(versions) == 1:
            for w in one_left:
                s = entity.find(w)
                if s != -1:
                    return self.__comfirm_the_boundary(entity, f"{versions[0]},", 1)

            return self.__comfirm_the_boundary(entity, f",{versions[0]}", 1)

        if len(versions) == 2:
            return self.__comfirm_the_boundary(entity, f"{versions[0]},{versions[1]}", 2)

        if len(versions) >= 3:
            version_str = f"{versions[0]},{versions[-1]}"
            return self.__comfirm_the_boundary(entity, version_str, 3)

    def __comfirm_the_boundary(self, entity: str, versions: str, versions_size: int):
        including_key_word = ['include', 'includ', 'through', '=']
        if versions_size < 1:
            return versions
        if versions_size == 1:
            for w in including_key_word:
                s = entity.find(w)
                if s != -1:
                    if versions.find(',') == 0:
                        return f"({versions}]"
                    else:
                        return f"[{versions})"
            return f"({versions})"
        if versions_size > 1:
            for w in including_key_word:
                s = entity.find(w)
                if s != -1:
                    return f"[{versions}]"
            return f"[{versions})"

    def __is_chinese(self, ch):
        if '\u4e00' <= ch <= '\u9fff':
            return True
        return False

    def __preprocess(self, entity: str) -> str:

        # special version convert to specific version (e.g 5.x->5.0)
        special_char_pattern = r'[/:*x]'
        special = re.compile(special_char_pattern, re.I)
        entity = special.sub('0', entity)

        # remove chinese char
        chinese_words = []
        for ch in entity:
            if self.__is_chinese(ch):
                chinese_words.append(ch)

        for word in chinese_words:
            entity = entity.replace(word, " ")
        return entity


class LangConvert:
    def convert(self, sentence: str):
        language = self.__find_file(sentence=sentence)
        language.extend(self.__find_language(sentence=sentence))
        return list(set(language))

    def __find_file(self, sentence: str) -> list:
        # define suffix to language map
        file_suffix_to_language = {
            ".c": "c",
            ".cpp": "c++",
            ".java": "java",
            ".py": "python",
            ".js": "javascript",
            ".go": "go",
            ".php": "php"
        }
        # define pattern
        file_pattern = r'\b\w+\.(c|java|go|py|cpp|js|php)\b'

        files = [match.group()
                 for match in re.finditer(file_pattern, sentence)]

        language = []
        for s in file_suffix_to_language:
            for file in files:
                if s in file:
                    language.append(file_suffix_to_language[s])
        return language

    def __find_language(self, sentence: str):
        language_pattern = r"\b(c\+{2}|c|python|javascript|go|java|php)\b"
        language = [match.group()
                    for match in re.finditer(language_pattern, sentence, re.I)]
        return language
