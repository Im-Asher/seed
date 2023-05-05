import re

class VersionConvert:
    version_pattern = r'\d+\.\d+(?:\.\d+)?(?:\w+|-\w+)?|\d+'

    def convert(self,entity:str,label:str):
        entity = entity.lower()
        if label == "VERL":
            return self.__convert_to_version_list(entity=entity)
        if label == "VERR":
            return self.__convert_to_versing_range(entity=entity)
        
    def __convert_to_version_list(self,entity:str):
        version_intervals = [(match.group(), match.start(), match.end())
                             for match in re.finditer(self.version_pattern, entity)]

        versions = [v[0] for v in version_intervals]

        versions = ','.join(versions)

        return versions
    
    def __convert_to_versing_range(self,entity:str):
        one_left = ['start', 'from', '>', '>=']
        one_right = ['prior', 'before', 'through',
                     'to', 'up', 'earlier', '<', '<=','below']

        # special version convert to specific version (e.g 5.x->5.0)
        special_char_pattern = r'[/:*x]'
        special = re.compile(special_char_pattern, re.I)

        version_intervals = [(match.group(), match.start(), match.end())
                             for match in re.finditer(self.version_pattern, entity)]

        versions = [special.sub('0', v[0]) for v in version_intervals]

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