import logging
import datetime
import logging.handlers

class customlogger :
    def __init__(self , name) :
        self.log = logging.getLogger(name)
        self.log.propagate = True
        self.formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s'
        )
        self.levels = {
            "DEBUG" : logging.DEBUG , 
            "INFO" : logging.INFO , 
            "WARNING" : logging.WARNING , 
            "ERROR" : logging.ERROR , 
            "CRITICAL" : logging.CRITICAL }
        
    def create_logger(self, file_name, mode, level):
        log = self.stream_handler(level)
        log = self.file_handler(file_name, mode, level)
        return log
    
    def stream_handler(self, level) :
        """
        level :
        > "DEBUG" : logging.DEBUG , 
        > "INFO" : logging.INFO , 
        > "WARNING" : logging.WARNING , 
        > "ERROR" : logging.ERROR , 
        > "CRITICAL" : logging.CRITICAL , 
        """
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(self.formatter)
        self.log.addHandler(streamHandler)
        self.log.setLevel(self.levels[level])
        return self.log
    
    def file_handler(self, file_name, mode, level) :
        """
        file_name : ~.txt / ~.log
        mode : "w" / "a"
        level :
        > "DEBUG" : logging.DEBUG , 
        > "INFO" : logging.INFO , 
        > "WARNING" : logging.WARNING , 
        > "ERROR" : logging.ERROR , 
        > "CRITICAL" : logging.CRITICAL , 
        """
        
        fileHandler = logging.FileHandler(file_name ,mode = mode)
        fileHandler.setFormatter(self.formatter)
        self.log.addHandler(fileHandler)
        self.log.setLevel(self.levels[level])
        return self.log
    def Rotating_filehandler(self, file_name , mode , level , backupCount , log_max_size ) :
        """
        file_name : ~.txt / ~.log
        mode : "w" / "a"
        backupCount : backup할 파일 개수
        log_max_size : 한 파일당 용량 최대
        level :
        > "DEBUG" : logging.DEBUG , 
        > "INFO" : logging.INFO , 
        > "WARNING" : logging.WARNING , 
        > "ERROR" : logging.ERROR , 
        > "CRITICAL" : logging.CRITICAL , 
        """
        
        fileHandler = logging.handlers.RotatingFileHandler(
            filename=file_name , 
            maxBytes=log_max_size,
            backupCount=backupCount,
            mode =  mode )
        fileHandler.setFormatter(self.formatter)
        self.log.addHandler(fileHandler)
        self.log.setLevel(self.levels[level])
        return self.log
    def timeRotate_handler(self , filename='./log.txt', 
                           when = "M" ,
                           level = "DEBUG" , 
                           backupCount= 4 , 
                           atTime = datetime.time(0, 0, 0),
                           interval = 1
                          ) :
        """
        file_name : 
        when : 저장 주기
        interval : 저장 주기에서 어떤 간격으로 저장할지 
        backupCount : 5
        atTime : datetime.time(0, 0, 0)
        """
        fileHandler = logging.handlers.TimedRotatingFileHandler(
            filename= filename, 
            when = when ,  # W0
            backupCount= backupCount , 
            interval = interval , 
            atTime=atTime )
        fileHandler.setFormatter(self.formatter)
        self.log.addHandler(fileHandler)
        self.log.setLevel(self.levels[level])
        return self.log