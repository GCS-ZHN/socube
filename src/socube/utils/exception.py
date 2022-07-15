# MIT License
#
# Copyright (c) 2022 Zhang.H.N
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from types import TracebackType
from typing import Type, TypeVar

from .context import ContextManagerBase
from .mail import MailService
from .logging import log

import traceback
import signal

__all__ = ["ExceptionManager"]

ExceptionManagerType = TypeVar('ExceptionManagerType',
                               bound="ExceptionManager")


class ExceptionManager(ContextManagerBase):
    """
    A context manager to deal with exception and notify users if it is required.
    Any exception occurred within this context will be catched and send notification
    email to user if mailService is not equal to 'None'. You use it as following.

    Parameters
    ----------
    mail_service : MailService
        A MailService object used to send email to users, If None, no email will be sent
    is_continue : bool
        If continue to run this program when an error occurred.
    ignore_keyboard_interrupt : bool
        If true, it will ignore keyboard interrupt signal, such as "Ctrl + C". It prevent
        quit program by pressing keyboard unexpectedly. But it also is a shortcoming
        when you want to terminate program.

    Examples
    ----------
    >>> mailService = MailService(...)
    >>> with ExceptionManager(mailService)
    >>>     a = 1/0  # will throw ZeroDivideError
    """

    def __init__(self,
                 mail_service: MailService = None,
                 is_continue: bool = False,
                 ignore_keyboard_interrupt: bool = False,
                 ignore_normal_system_exit: bool = True) -> None:
        self._mail_service = mail_service
        self._continue = is_continue
        self._ignore_keyboard_interrupt = ignore_keyboard_interrupt
        self._ignore_normal_system_exit = ignore_normal_system_exit
        self._normal_text = None

    def setNormalInfo(self, text: str) -> None:
        """
        Set the notification email when quiting context without any exception occurred.
        This can be used to send a processing result. Result should as short as possible.

        Args:
            text         email main text

        >>> with ExceptionManager(mailService) as em:
        >>>     # Here are some processing codes
        >>>     em.setNormalInfo("Prcoessing results")
        """
        self._normal_text = text

    def setMailService(self, mail_service: MailService) -> None:
        """
        Set the mail service object.
        """
        self._mail_service = mail_service

    def __enter__(self: ExceptionManagerType) -> ExceptionManagerType:
        if self._ignore_keyboard_interrupt:
            signal.signal(
                signal.SIGINT,
                lambda s, f: log("Exception", "Ignore sginal {s}: Ctrl + C"))
        return self

    def __exit__(self, exc_type: Type[Exception], exc_obj: Exception,
                 exc_trace: TracebackType) -> bool:
        try:
            assert exc_obj is None or isinstance(
                exc_obj, exc_type), "excVal must be a instance of excType!"

            if self._ignore_keyboard_interrupt:
                signal.signal(signal.SIGINT, signal.default_int_handler)

            skipped = self._ignore_normal_system_exit and isinstance(
                exc_obj, SystemExit) and exc_obj.code == 0

            if isinstance(self._mail_service, MailService):
                with self._mail_service as mail:
                    if skipped:
                        log(
                            "Exception",
                            "Ignore sginal SystemExit: exit code {code}".
                            format(code=exc_obj.code))
                    elif exc_type is not None:
                        mail.sendSelf(
                            "Program notification", "".join(
                                traceback.format_exception(
                                    exc_type, exc_obj, exc_trace, 1)))
                    elif self._normal_text is not None:
                        mail.sendSelf("Program notification",
                                      self._normal_text)
            # if not continue, error will raise again so it is not necessary to print it ourself
            if self._continue and exc_obj and not skipped:
                traceback.print_exception(exc_type, exc_obj, exc_trace)
        except Exception as e:
            log("Exception",
                f"A unexpected error occured while exiting context:{e}")
        finally:
            return self._continue
