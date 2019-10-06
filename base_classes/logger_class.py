import sys
import datetime


class Logger:
    """
    Class used to overwrite sys.stdout to redirect stdout stream additionally to log file.
    Class saves sys.stdout as one of its variables, which allows to normally handle "print()" and
    "sys.stdout.write()" commands and send their content to terminal.
    Class is automatically adding timestamp to each printed and saved to log file message.
    """

    def __init__(self, log_file):
        """
        :param log_file: Path to the file in which stdout stream will be saved.
        """
        self.terminal = sys.stdout  # Adding sys.stdout to Class, which allows to normally handle "print()" commands
        self.log = log_file

        # Creating empty log file with content of stdout stream
        with open(self.log, 'w') as log_file:
            log_file.write("Logs started at {} \n".format(
                str(datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))))

    def write(self, message):
        """
        Method called by using print(message) or sys.stdout.write(message).
        At the beginning of every message timestamp is automatically added.

        :param message: String with argument from print() or sys.stdout.write()
        :return:
        """

        # Sending message to terminal with sys.stdout, just as print() or sys.stdout.write() would do.
        if message != "\n":
            # If message is not endline symbol, add timestamp at the beginning of message.
            time_stamp = str(datetime.datetime.now())
            self.terminal.write(time_stamp + ": " + message)
        else:
            self.terminal.write(message)

        # Saving message to log file
        try:
            with open(self.log, 'a') as log_file:
                if message != "\n":
                    # If message is not endline symbol, add timestamp at the beginning of message.
                    time_stamp = str(datetime.datetime.now())
                    log_file.write(time_stamp + ": " + message)
                else:
                    log_file.write(message)

        except Exception as inst:   # If it is not possible to open log file, write error communicate to terminal
            self.terminal.write(str(inst))

    def flush(self):
        """
        This flush method is needed for python 3 compatibility.
        This handles the flush command by doing nothing.
        You might want to specify some extra behavior here.
        :return:
        """
        pass
