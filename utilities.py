"""
General utilities.
"""


from sage.all import parent, ZZ


def balance(e, q=None):
    """ Return a representation of `e` with elements balanced between `-q/2` and `q/2`
    :param e: a vector, polynomial or scalar
    :param q: optional modulus, if not present this function tries to recover it from `e`
    :returns: a vector, polynomial or scalar over/in the integers
    """
    try:
        p = parent(e).change_ring(ZZ)
        return p([balance(e_, q=q) for e_ in e])
    except (TypeError, AttributeError):
        if q is None:
            try:
                q = parent(e).order()
            except AttributeError:
                q = parent(e).base_ring().order()
        e = ZZ(e)
        e = e % q
        return ZZ(e-q) if e > q//2 else ZZ(e)


class Domain:
    def __init__(self, a, b, definition):
        """ Given floats `a < b`, it returns a class describing a domain of
        `definition` equally spaced points in [a, b], extremities included.

        :params a:          float
        :params b:          float, larger than a
        :params definition: int, number of points
        """

        if a >= b:
            raise ValueError("`a` must be less than `b`")

        if int(definition) != definition:
            raise ValueError("`definition` must be integer")

        self.a = float(a)
        self.b = float(b)
        self.h = self.b - self.a
        self.definition = int(definition)
        self.domain = list(self.a + _ * self.h/(self.definition-1)
                           for _ in range(self.definition))

    def grid(self, structure="generator"):
        """ Returns a data structure for he points in the domain.
        """

        if structure == "dict":
            return dict(zip(self.domain, [0] * self.definition))
        if structure == "list":
            return list(_ for _ in self.domain)
        return (_ for _ in self.domain)

    @staticmethod
    def _binarySearch(data, val):
        """ Binary searchf or closest value to `val` in `data`.
        Only works if each element is present once in list.
        Taken from https://stackoverflow.com/a/27337924.
        """
        val = float(val)
        lo, hi = 0, len(data) - 1
        best_ind = lo
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if data[mid] < val:
                lo = mid + 1
            elif data[mid] > val:
                hi = mid - 1
            else:
                best_ind = mid
                break
            # check if data[mid] is closer to val than data[best_ind]
            if abs(data[mid] - val) < abs(data[best_ind] - val):
                best_ind = mid
        return best_ind

    def round(self, x, return_index=False):
        """ Given a float x, find the closest float in the domain.
        """
        idx = self._binarySearch(self.domain, x)
        if return_index:
            return idx
        return self.domain[idx]


def my_hsv_to_rgb(hsv):
    """ Convert HSV color representation into hex RGB code.
    """
    from matplotlib.colors import hsv_to_rgb
    rgb_array = hsv_to_rgb(hsv)
    return "#" + "".join(map(lambda x: "%02x" % int(255*x), rgb_array))


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    Black = '\u001b[30m'
    Red = '\u001b[31m'
    Green = '\u001b[32m'
    Yellow = '\u001b[33m'
    Blue = '\u001b[34m'
    Magenta = '\u001b[35m'
    Cyan = '\u001b[36m'
    White = '\u001b[37m'

    BrightBlack = '\u001b[30;1m'
    BrightRed = '\u001b[31;1m'
    BrightGreen = '\u001b[32;1m'
    BrightYellow = '\u001b[33;1m'
    BrightBlue = '\u001b[34;1m'
    BrightMagenta = '\u001b[35;1m'
    BrightCyan = '\u001b[36;1m'
    BrightWhite = '\u001b[37;1m'

    BackgroundBlack = '\u001b[40m'
    BackgroundRed = '\u001b[41m'
    BackgroundGreen = '\u001b[42m'
    BackgroundYellow = '\u001b[43m'
    BackgroundBlue = '\u001b[44m'
    BackgroundMagenta = '\u001b[45m'
    BackgroundCyan = '\u001b[46m'
    BackgroundWhite = '\u001b[47m'

    BackgroundBrightBlack = '\u001b[40;1m'
    BackgroundBrightRed = '\u001b[41;1m'
    BackgroundBrightGreen = '\u001b[42;1m'
    BackgroundBrightYellow = '\u001b[43;1m'
    BackgroundBrightBlue = '\u001b[44;1m'
    BackgroundBrightMagenta = '\u001b[45;1m'
    BackgroundBrightCyan = '\u001b[46;1m'
    BackgroundBrightWhite = '\u001b[47;1m'

    Bold = '\u001b[1m'
    Underline = '\u001b[4m'
    Reversed = '\u001b[7m'
