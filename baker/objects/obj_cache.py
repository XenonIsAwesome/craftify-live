class ObjectCache:
    instance = None
    cache = {}

    @staticmethod
    def get_instance():
        if ObjectCache.instance is None:
            ObjectCache.instance = ObjectCache()
        return ObjectCache.instance

    def __init__(self):
        ObjectCache.cache = {}

    def __setattr__(self, name, value):
        ObjectCache.cache[name] = value


    def __delattr__(self, name):
        del ObjectCache.cache[name]
    
    def __getattr__(self, name):
        return ObjectCache.cache[name]

    def __contains__(self, key):
        return key in ObjectCache.cache

    def __iter__(self):
        return iter(ObjectCache.cache)

    def __len__(self):
        return len(ObjectCache.cache)

    def __repr__(self):
        return repr(ObjectCache.cache)