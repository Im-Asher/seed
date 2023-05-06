import re
from commom import is_chinese

class VersionConvert:
    __version_pattern = r'\b\d+\.\d+(?:\.\d+)*(?:\w+|-\w+)?|\d+\b'

    def convert(self,entity:str,label:str):
        entity = entity.lower()

        entity = self.__preprocess(entity)

        if label == "VERL":
            return self.__convert_to_version_list(entity=entity)
        if label == "VERR":
            return self.__convert_to_version_range(entity=entity)
        
    def __convert_to_version_list(self,entity:str):
        version_intervals = [(match.group(), match.start(), match.end())
                             for match in re.finditer(self.__version_pattern, entity)]

        versions = [v[0] for v in version_intervals]

        versions = ','.join(versions)

        return versions
    
    def __convert_to_version_range(self,entity:str):
        one_left = ['start', 'from', '>', '>=']
        one_right = ['prior', 'before', 'through',
                     'to', 'up', 'earlier', '<', '<=','below']

        versions = [match.group() for match in re.finditer(self.__version_pattern, entity)]

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
        
    def __preprocess(self,entity:str)->str:

        # special version convert to specific version (e.g 5.x->5.0)
        special_char_pattern = r'[/:*x]'
        special = re.compile(special_char_pattern, re.I)
        entity = special.sub('0',entity)

        # remove chinese char
        chinese_words = []
        for ch in entity:
            if is_chinese(ch):
                chinese_words.append(ch)
        
        for word in chinese_words:
            entity = entity.replace(word," ")
        return entity

if __name__=="__main__":
    vc =  VersionConvert()
    s = vc.convert("2.1和3.2","VERR")
    print(s)