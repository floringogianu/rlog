import sys, traceback
from termcolor import colored as clr


def print_fancy_err(err, issue=None, fix=None):
    _, _, exc_tb = sys.exc_info()
    tb = traceback.extract_tb(exc_tb)
    stack = traceback.extract_stack()
    what = "what: {}: {}".format(
        type(err).__name__, clr(str(err), attrs=["bold"])
    )
    fpath, line_no, _, code = tuple(tb[-1])
    where = 'where: "{0}", line {1}\n\n\t| {1} >   {2}'.format(
        fpath, line_no, code
    )
    fpath, line_no, _, code = tuple(stack[1])
    entry_point = 'from: "{0}", line {1}\n\n\t| {1} >   {2}'.format(
        fpath, line_no, code
    )
    msg = "\n\t{}\n\t{}\n\n\t{}\n".format(what, where, entry_point)
    if issue is not None:
        msg += "\n\tCause: {}.".format(clr(issue, "red"))
    if fix is not None:
        hint = clr("Hint", attrs=["underline"])
        msg += "\n\t{}: {}.".format(hint, clr(fix, "green"))
    msg += "\n"
    print(msg)
