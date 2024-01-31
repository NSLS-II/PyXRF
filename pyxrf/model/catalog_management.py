class CatalogInfo:
    def __init__(self):
        self._name = None

    @property
    def name(self):
        return self._name

    def set_name(self, name):
        self._name = name


catalog_info = CatalogInfo()


def get_catalog(catalog_name):
    from tiled.client import from_uri

    c = from_uri("https://tiled.nsls2.bnl.gov")
    return c[catalog_name.lower()]["raw"]
