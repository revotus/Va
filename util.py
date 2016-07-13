"""Utilities for clustering."""
import logging
import weakref
from kitchen.text.converters import to_bytes


def logged(cls):
    """Decorate class to create logger automatically.

    Instantiates a logger attribute for the class labeld after the module and
    class. Also writes a logging message everytime an instance of the class is
    initialized.

    Args:
        cls: Any class

    Returns:
       The input class with a logger attribute and a new __init__ method that
       writes a logging message every time an instance of the class is
       initialized and then calls the original __init__.
    """
    cls.logger = logging.getLogger('.'.join([cls.__module__, cls.__name__]))

    orig_init = cls.__init__

    def __init__(self, *args, **kwds):
        """Write debug instantiation message to log then inits cls."""
        self.logger.debug('Creating instance of %s with\nargs:\t%s \n'
                          'kwds:\t%s...', cls.__name__,
                          to_bytes(args), to_bytes(kwds))
        orig_init(self, *args, **kwds)

    cls.__init__ = __init__
    return cls


class AutoLabeler(type):

    """Metaclass that auto-labels parameters.

    Metaclass that creates classes that auto-assign parameter keys
    to 'label' attribute of said parameter for every Labeled
    parameter in instance of created class's __dict__.

    e.g. for
    class Foo(object):
        __metaclass__ = AutoLabeler()
        bar = Labeled()

    Foo will have class Labeled parameter with Foo.bar.label = 'bar'
    """

    def __new__(mcs, clsname, bases, attrs):
        """Create new metaclass with auto-labeled params."""
        for key, value in attrs.items():
            if isinstance(value, Labeled):
                value.label = key
        return super(AutoLabeler, mcs).__new__(mcs, clsname, bases, attrs)


@logged
class Labeled(object):

    """Inherited by descriptors for labeling.

    Attributes:
        label (str): The label of the particular descriptor.
    """

    def __init__(self, label=None, **kwds):
        """Initialized with label."""
        self.label = label
        super(Labeled, self).__init__(**kwds)

    def __repr__(self):
        """"Return label if set, otherwise call super."""
        if self.label is None:
            return super(Labeled, self).__repr__()
        return self.label


@logged
class Descriptor(Labeled):

    """ Descriptor that looks up value by using instance's __dict__."""

    def __init__(self, **kwds):
        """Init super."""
        super(Descriptor, self).__init__(**kwds)

    def __get__(self, instance, owner):
        """Get descriptor param from instance's __dict__."""
        if instance is None:
            return self
        try:
            return instance.__dict__[self]
        except KeyError:
            raise AttributeError("'%s' object has no attribute '%s'" %
                                 (owner.__name__, self.label))

    def __set__(self, instance, value):
        """Set descriptor param in instance's __dict__ to value."""
        instance.__dict__[self] = value

    def __delete__(self, instance):
        """Nulete decsriptor param from instance's __dict__."""
        try:
            del instance.__dict__[self]
        except KeyError:
            pass


@logged
class SubjectMixin(object):

    """Template for creating the subject in the observer pattern."""

    def __init__(self, observers=None, **kwds):
        """Init with WeakSet for observers."""
        self._observers = weakref.WeakSet()
        super(SubjectMixin, self).__init__(**kwds)

        if observers:
            self._observers.update(observers)

    def notify(self, instance):
        """Notify observers of change in SubjectMixin."""
        for observer in self._observers:
            observer.update(instance)

    def add_observer(self, observer):
        """Add observer to observers set."""
        self._observers.add(observer)

    def remove_observer(self, observer):
        """Remove observer from observers set."""
        if observer in self._observers:
            self._observers.remove(observer)


@logged
class SubjDesc(Descriptor, SubjectMixin):

    """Descriptor that notifies observers on update."""

    def __init__(self, **kwds):
        """Init."""
        super(SubjDesc, self).__init__(**kwds)

    def __set__(self, instance, value):
        """Set value then notify observers."""
        super(SubjDesc, self).__set__(instance, value)
        self.notify(instance)

    def __delete__(self, instance):
        """Delete instance's descriptor param then notify observers."""
        super(SubjDesc, self).__delete__(instance)
        self.notify(instance)


@logged
class NulObsDesc(Descriptor):

    """Descriptor that is set to None on update."""

    def __init__(self, **kwds):
        """Init."""
        super(NulObsDesc, self).__init__(**kwds)

    def update(self, instance):
        """Set instance's descriptor param to None."""
        self.__set__(instance, None)


@logged
class SubjNulObsDesc(SubjDesc, NulObsDesc):

    """Subject and observer descriptor that sets value to None on update."""

    def __init__(self, **kwds):
        """Init."""
        super(SubjNulObsDesc, self).__init__(**kwds)


@logged
class TypedDesc(Descriptor):

    """Descriptor that checks type of value and type of keys if applicable."""

    def __init__(self, typ, subtyp=None, **kwds):
        """Init with typ and subtyp."""
        self.typ = typ
        self.subtyp = subtyp
        super(TypedDesc, self).__init__(**kwds)

    def __set__(self, instance, value):
        """Check types before setting."""
        if not isinstance(value, self.typ) and value is not None:
            raise TypeError('%s must be %s not %s' %
                            (self.label, self.typ, type(value)))
        if self.subtyp is not None and value:
            for item in value:
                if not isinstance(item, self.subtyp) and item is not None:
                    raise TypeError('%s items must be %s not %s' %
                                    (self.label, self.subtyp, type(item)))
        super(TypedDesc, self).__set__(instance, value)


@logged
class TypSubjDesc(TypedDesc, SubjDesc):

    """Subject descriptor that is also typed."""

    def __init__(self, **kwds):
        """Init."""
        super(TypSubjDesc, self).__init__(**kwds)

    def __set__(self, instance, value):
        """Set with type checks and subj protocols."""
        super(TypSubjDesc, self).__set__(instance, value)


@logged
class TypNulObsDesc(TypedDesc, NulObsDesc):

    """A deletion observer descriptor that is also typed."""

    def __init__(self, **kwds):
        """Init."""
        super(TypNulObsDesc, self).__init__(**kwds)


@logged
class TypSubjNulObsDesc(TypSubjDesc, TypNulObsDesc):

    """Typed descriptor that is subject and observer.

    A typed descriptor that is both observer of some subject and subject
    for other observer(s).
    """

    def __init__(self, **kwds):
        """Init."""
        super(TypSubjNulObsDesc, self).__init__(**kwds)


@logged
class CacheDesc(Labeled):

    """Caching descriptor suitable for unhashable classes.

    Alternative datastructure to 'Descriptor'.

    Object id is used as key in caching dictionary to allow unhashable
    classes to implement this.  The values in the data dictionary are tuples
    consisting of the intended value and a weakref to the object. The weakref
    is included to catch the case that an object is destroyed and a new object
    with the same id is created.

    Attributes:
        data (dict):  A dictionary for looking up an object's descriptor value.
    """

    def __init__(self, label=None, **kwds):
        """Init with WeakKeyDictionary."""
        self.instances = weakref.WeakKeyDictionary()
        # self.data = {}
        super(CacheDesc, self).__init__(label=label, **kwds)

    def __get__(self, instance, owner):
        """Look up object's value by id.

        Uses id to look up value and weakref for object. If id is found, value
        and weakref are returned. If weakref is None, a new object with the
        same id as a deleted object has been created, so id is deleted. Then,
        """
        if instance is None:
            return self
        inst_id = id(instance)
        try:
            value, wr = self.instances[inst_id]
        except KeyError:
            raise AttributeError("'%s' object has no attribute '%s'" %
                                 (owner.__name__, self.label))
        else:
            if wr() is None:
                del self.instances[inst_id]
                raise AttributeError("'%s' object has no attribute '%s'" %
                                     (owner.__name__, self.label))
        return value

    def __set__(self, instance, value):
        """Set instance's descriptor to value and a weakref to the instance."""
        self.instances[id(instance)] = (value, weakref.ref(instance))

    def __delete__(self, instance):
        """Nulete instance's id key from descriptor dictionary."""
        try:
            del self.instances[id(instance)]
        except KeyError:
            pass


@logged
class DefDesc(Descriptor):

    """Descriptor that reverts to a default when no value is set.

    Descriptor that reverts to a default when no value is set for descriptor
    and descriptor is called. Default can be a variable, instance method, or
    other callable.

    Attributes:
        default: An attribute, variable, or method that is used or called as a
            default when an object has no value set for the descriptor.
    """

    def __init__(self, default, **kwds):
        """Init."""
        self.default = default
        super(DefDesc, self).__init__(**kwds)

    def __get__(self, instance, cls):
        """Get value. If no value set, get from _val_from_def.

        First try regular descriptor access, if no value is found, get
        value using _val_from_def method.
        """
        try:
            return super(DefDesc, self).__get__(instance, cls)
        except AttributeError:
            value = self._val_from_def(instance)
            super(DefDesc, self).__set__(instance, value)
            return value

    def _val_from_def(self, instance):
        """Return value using descriptor's default.

        Set default as either attribute of instance or variable.  Then,
        decides if default is callable and calls it and returns it if so,
        otherwise returns default.
        """
        default = getattr(instance, str(self.default), self.default)
        if callable(default):
            self.logger.debug('Looking up %s for %s...', self.label, instance)
            value = default()
            try:
                self.logger.debug('Found %d %s for %s',
                                  len(value), self.label, instance)
            except TypeError:
                self.logger.debug('Found %d for %s', value, instance)
        else:
            value = default
        return value


@logged
class DefObsDesc(DefDesc):

    """Observer attribute that deletes it's value on notification.

    Observer attribute that deletes it's value on notification and retrieves
    a new value from default when next accessed.
    """

    def __init__(self, default, **kwds):
        """Init."""
        super(DefObsDesc, self).__init__(default=default, **kwds)

    def update(self, instance):
        """Nulete value on update."""
        if self in instance.__dict__:
            self.__delete__(instance)


@logged
class TypDefObsDesc(TypedDesc, DefObsDesc):

    """A default obsever descriptor that is also typed."""

    def __init__(self, typ, subtyp=None, default=None, **kwds):
        """Init."""
        super(TypDefObsDesc, self).__init__(typ=typ, subtyp=subtyp,
                                            default=default, **kwds)


class DefSubjDesc(DefDesc, SubjDesc):

    """Subject descriptor that accesses default if not set."""

    def __init__(self, default, **kwds):
        """Init."""
        super(DefSubjDesc, self).__init__(default=default, **kwds)


class TypDefSubjDesc(TypSubjDesc, DefSubjDesc):

    """A typed subject descriptor that uses a default when not set."""

    def __init__(self, typ, subtyp=None, default=None, **kwds):
        """Init."""
        super(TypDefSubjDesc, self).__init__(typ=typ, subtyp=subtyp,
                                             default=default, **kwds)


class DefSubjObsDesc(DefSubjDesc, DefObsDesc):

    """Subject and observer descriptor that accesses defualt if not set."""

    def __init__(self, default, observers=None, **kwds):
        """Init."""
        super(DefSubjObsDesc, self).__init__(default=default,
                                             observers=observers, **kwds)


class TypDefSubjObsDesc(TypSubjDesc, TypDefObsDesc):

    """Typed descriptor that is subject and observer.

    A typed descriptor that is both observer of some subject and subject
    for other observer(s).
    """

    def __init__(self, typ, subtyp=None, default=None, observers=None, **kwds):
        """Init."""
        super(TypDefSubjObsDesc, self).__init__(typ=typ, subtyp=subtyp,
                                                default=default,
                                                observers=observers, **kwds)
