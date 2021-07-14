import sys
import pkgutil
import inspect
import importlib
from collections import OrderedDict


def get_combined_candidtates(builtin_candidates, thirdparty_candidates):
    candidates = builtin_candidates.copy()
    candidates.update(thirdparty_candidates.components)
    return candidates


def get_combined_fe_candidtates(builtin_candidates, thirdparty_candidates, tran_types=None):
    candidates = builtin_candidates.copy()
    candidates.update(thirdparty_candidates.components)
    if tran_types:
        for key in thirdparty_candidates.components:
            tran_types.append(thirdparty_candidates.components[key].type)
        return candidates, tran_types
    else:
        return candidates


def find_components(package, directory, base_class):
    components = OrderedDict()

    for module_loader, module_name, ispkg in pkgutil.iter_modules([directory]):
        full_module_name = "%s.%s" % (package, module_name)
        if full_module_name not in sys.modules and not ispkg:
            module = importlib.import_module(full_module_name)

            for member_name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, base_class) and \
                        obj != base_class:
                    # Keep in mind that this only instantiates the ensemble_wrapper,
                    # but not the real target classifier
                    classifier = obj
                    components[module_name] = classifier

    return components


class ThirdPartyComponents(object):
    def __init__(self, base_class):
        self.base_class = base_class
        self.components = OrderedDict()

    def add_component(self, obj):
        if inspect.isclass(obj) and self.base_class in obj.__bases__:
            name = obj.__name__
            classifier = obj
        else:
            raise TypeError('add_component works only with a subclass of %s' %
                            str(self.base_class))

        properties = set(classifier.get_properties())
        should_be_there = {'shortname', 'name', 'handles_regression',
                           'handles_classification', 'handles_multiclass',
                           'handles_multilabel', 'is_deterministic',
                           'input', 'output'}
        for property in properties:
            if property not in should_be_there:
                raise ValueError('Property %s must not be specified for '
                                 'algorithm %s. Only the following properties '
                                 'can be specified: %s' %
                                 (property, name, str(should_be_there)))
        for property in should_be_there:
            if property not in properties:
                raise ValueError('Property %s not specified for algorithm %s')

        self.components[name] = classifier
