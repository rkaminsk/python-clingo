'''
Module providing functions and classes to implement applications based on
clingo.
'''

from typing import Callable, Iterable, Sequence
from abc import ABCMeta, abstractmethod

from .core import MessageCode
from .solving import Model
from .control import Control

class Flag:
    '''
    Helper object to parse command-line flags.

    Parameters
    ----------
    value : bool=False
        The initial value of the flag.
    '''
    def __init__(self, value: bool=False):
        self.flag = value

    flag: bool
    '''
    The value of the flag.
    '''

class ApplicationOptions(metaclass=ABCMeta):
    '''
    Object to add custom options to a clingo based application.
    '''
    def add(self, group: str, option: str, description: str, parser: Callable[[str], bool],
            multi: bool=False, argument: str=None) -> None:
        '''
        Add an option that is processed with a custom parser.

        Parameters
        ----------
        group : str
            Options are grouped into sections as given by this string.
        option : str
            Parameter option specifies the name(s) of the option. For example,
            `"ping,p"` adds the short option `-p` and its long form `--ping`. It is
            also possible to associate an option with a help level by adding `",@l"` to
            the option specification. Options with a level greater than zero are only
            shown if the argument to help is greater or equal to `l`.
        description : str
            The description of the option shown in the help output.
        parser : Callable[[str],bool]
            An option parser is a function that takes a string as input and returns
            true or false depending on whether the option was parsed successively.
        multi : bool=False
            Whether the option can appear multiple times on the command-line.
        argument : str=None
            Optional string to change the value name in the generated help.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            An error is raised if an option with the same name already exists.

        Notes
        -----
        The parser also has to take care of storing the semantic value of the option
        somewhere.
        '''

    def add_flag(self, group: str, option: str, description: str, target: Flag) -> None:
        '''
        Add an option that is a simple flag.

        This function is similar to `ApplicationOptions.add` but simpler because
        it only supports flags, which do not have values. Note that the target
        parameter must be of type Flag, which is set to true if the flag is passed on
        the command line.

        Parameters
        ----------
        group : str
            Options are grouped into sections as given by this string.
        option : str
            Same as for `ApplicationOptions.add`.
        description : str
            The description of the option shown in the help output.
        target : Flag
            The object that receives the value.

        Returns
        -------
        None
        '''

class Application(metaclass=ABCMeta):
    '''
    Interface that has to be implemented to customize clingo.

    Attributes
    ----------
    program_name: str = 'clingo'
        Optional program name to be used in the help output.

    message_limit: int = 20
        Maximum number of messages passed to the logger.
    '''

    @abstractmethod
    def main(self, control: Control, files: Sequence[str]) -> None:
        '''
        Function to replace clingo's default main function.

        Parameters
        ----------
        control : Control
            The main control object.
        files : Sequence[str]
            The files passed to clingo_main.

        Returns
        -------
        None
        '''

    def register_options(self, options: ApplicationOptions) -> None:
        '''
        Function to register custom options.

        Parameters
        ----------
        options : ApplicationOptions
            Object to register additional options

        Returns
        -------
        None
        '''

    def validate_options(self) -> bool:
        '''
        Function to validate custom options.

        This function should return false or throw an exception if option
        validation fails.

        Returns
        -------
        bool
        '''

    def print_model(self, model: Model, printer: Callable[[], None]) -> None:
        '''
        Function to print additional information when the text output is used.

        Parameters
        ----------
        model : model
            The current model
        printer : Callable[[], None]
            The default printer as used in clingo.

        Returns
        -------
        None
        '''

    def logger(self, code: MessageCode, message: str) -> None:
        '''
        Function to intercept messages normally printed to standard error.

        By default, messages are printed to stdandard error.

        Parameters
        ----------
        code : MessageCode
            The message code.
        message : str
            The message string.

        Returns
        -------
        None

        Notes
        -----
        This function should not raise exceptions.
        '''

def clingo_main(application: Application, files: Iterable[str]=[]) -> int:
    '''
    Runs the given application using clingo's default output and signal handling.

    The application can overwrite clingo's default behaviour by registering
    additional options and overriding its default main function.

    Parameters
    ----------
    application : Application
        The Application object (see notes).
    files : Iterable[str]
        The files to pass to the main function of the application.

    Returns
    -------
    int
        The exit code of the application.

    Notes
    -----
    The main function of the `Application` interface has to be implemented. All
    other members are optional.

    Examples
    --------
    The following example reproduces the default clingo application:

        import sys
        import clingo

        class Application(clingo.Application):
            def __init__(self, name):
                self.program_name = name

            def main(self, ctl, files):
                if len(files) > 0:
                    for f in files:
                        ctl.load(f)
                else:
                    ctl.load("-")
                ctl.ground([("base", [])])
                ctl.solve()

        clingo.clingo_main(Application(sys.argv[0]), sys.argv[1:])
    '''
    # pylint: disable=dangerous-default-value,unused-argument,unnecessary-pass
    pass
