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

import smtplib
import os.path as path
from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formataddr
from typing import Dict, List, TypeVar
from .context import ContextManagerBase
from .logging import log

__doc__ = """This module provides some mail service tools"""
__all__ = ["MailService"]

MailServiceType = TypeVar("MailServiceType", bound="MailService")


class MailService(ContextManagerBase):
    """
    Provide basic SMTP mail service for mail sending. SMTP is the
    abbreviation of Simple Mail Transfer Protocol. For detail, please
    search it in google or bing. Context manager is supported.
    You can use it as following.

    Parameters
    ----------
    sender: str
        email address as a sender
    passwd: str
        sender's password for SMTP service. For some SMTP service vendors,
        this password is the same as password used to visit mail website,
        such as Zhejiang University Mail Service. But for other vendors like
        QQ Mail Service, you need use independent password for SMTP service.
        For detail, you need visit related mail service vendors.
    server: str
        the domain of SMTP server. such as 'smtp.zju.edu.cn','smtp.qq.com',
        which provided by the mail service vendors.
    port: int
        web port for SMTP service provided by service vendors.
    nickname: str
        the nickname of sender when sending a mail
    ssl: bool
        If True, use SSL for security.
    verbose: bool
        If True, mail service will run in background without any output info.


    >>> with MailService("account@gmail.com", "password", "smtp.gmail.com", 465) as ms
    >>>     ms.sendText(...)

    """

    def __init__(self,
                 sender: str,
                 passwd: str,
                 server: str,
                 port: int,
                 nickname: str = "SoCube",
                 ssl: bool = True,
                 verbose: bool = False) -> None:
        """
        Args:
            sender     email address as a sender
            passwd     sender's password for SMTP service. For some SMTP service vendors,
                       this password is the same as password used to visit mail website,
                       such as Zhejiang University Mail Service. But for other vendors like
                       QQ Mail Service, you need use independent password for SMTP service.
                       For detail, you need visit related mail service vendors.
            server     the domain of SMTP server. such as 'smtp.zju.edu.cn','smtp.qq.com',
                       which provided by the mail service vendors.
            port       web port for SMTP service provided by service vendors.
            nickName   the nickname of sender when sending a mail
            ssl        If True, use SSL for security.
            verbose    If True, mail service will run in background without any output info.
        """
        self._sender = sender
        self._passwd = passwd
        self._status = False
        self._nickname = nickname
        self._verbose = verbose
        self._ssl = ssl
        self._server = server
        self._port = port

    def _log(self, msg, *args, **kwargs):
        """Internal log function"""
        if not self._verbose:
            log(self.__class__.__name__, msg, *args, **kwargs)

    @property
    def status(self) -> bool:
        """Is service available"""
        return self._status

    def login(self) -> bool:
        """
        Login to enable service. After logining, you should use service
        before connection timeout.
        """
        if self._status:
            return
        try:
            if self._ssl:
                self._service = smtplib.SMTP_SSL(self._server, self._port)
            else:
                self._service = smtplib.SMTP(self._server, self._port)

            self._service.login(self._sender, self._passwd)
            self._status = True
        except Exception as e:
            self._log(f"Error '{e}' occurred while logining mail service")
        finally:
            return self._status

    def quit(self):
        """Quit to close service and release resources occuppied"""
        try:
            if self._status:
                self._service.quit()
        except Exception as e:
            self._log(f"Error '{e}' occurred while quiting mail service")
        finally:
            self._status = False

    def __enter__(self) -> MailServiceType:
        self.login()
        return self

    def __exit__(self, *args) -> bool:
        self.quit()
        return False

    # release resources before object destroyed
    __del__ = quit

    def sendText(self,
                 to_dict: Dict[str, str],
                 subject: str,
                 text: str,
                 attachments: List[str] = []) -> bool:
        """
        Send a plain text email with or without attachments

        Parameters
        ----------
        to_dict: Dict[str, str]
            a dict of email address and nickname.
        subject: str
            the subject of email.
        text: str
            email's main text. rich html format is not supported.
        attachments: List[str]
            a list of attachments' filename. attachments which are not existed
            will be skipped automatically

        Returns
        -------
        bool, True if sending succeed, False if sending failed.
        """
        if not self._status:
            self._log("Mail service is not available")
            return False
        try:
            msg = MIMEMultipart()
            msg.attach(MIMEText(text, "plain", "utf-8"))
            msg["From"] = formataddr([self._nickname, self._sender])
            msg["To"] = ",".join(
                [formataddr([addr, to_dict[addr]]) for addr in to_dict])
            msg["Subject"] = subject
            for attachfile in attachments:
                if not path.isfile(attachfile):
                    continue
                filename = path.basename(attachfile)
                with open(attachfile, mode="rb") as attachInput:
                    mine = MIMEBase('application',
                                    'octet-stream',
                                    filename=filename)
                    mine.add_header('Content-Disposition',
                                    'attachment',
                                    filename=filename)
                    mine.set_payload(attachInput.read())
                    encoders.encode_base64(mine)
                    msg.attach(mine)
            self._service.sendmail(self._sender, to_dict.keys(),
                                   msg.as_string())
            self._log("Sending mail successful")
            return True
        except Exception as e:
            self._log(f"Error '{e}' occurred while sending mail")

    def sendSelf(self,
                 subject: str,
                 text: str,
                 attachments: List[str] = []) -> bool:
        """
        Send a plain text mail without any attachment to sendor himself.

        Parameters
        ----------
        subject: str
            the subject of email.
        text: str
            email's main text. rich html format is not supported.
        attachments: List[str]
            a list of attachments' filename. attachments which are not existed
            will be skipped automatically

        Returns
        -------
        bool, True if sending succeed, False if sending failed.
        """
        return self.sendText({self._sender: self._nickname}, subject, text,
                             attachments)
