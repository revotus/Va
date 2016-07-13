def strip_accents_unicode(s):
    """Transform accentuated unicode symbols into their simple counterpart.

    Warning: the python-level loop and join operations make this
    implementation 20 times slower than the strip_accents_ascii basic
    normalization.
    See also
    --------
    strip_accents_ascii
    Remove accentuated char for any unicode symbol that has a direct
    ASCII equivalent.
    """
    return ''.join([c for c in unicodedata.normalize('NFKD', s)
                    if not unicodedata.combining(c)])


def strip_accents_ascii(s):
    """Transform accentuated unicode symbols into ascii or nothing
    Warning: this solution is only suited for languages that have a direct
    transliteration to ASCII symbols.
    See also
    --------
    strip_accents_unicode
    Remove accentuated char for any unicode symbol.
    """
    nkfd_form = unicodedata.normalize('NFKD', s)
    return nkfd_form.encode('ASCII', 'ignore').decode('ASCII')
