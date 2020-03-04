import numpy as np

from PyQt4.uic import loadUiType
from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import *
from PyQt4.QtCore import QObject,pyqtSignal

import pyqtgraph as pg
import pyqtgraph
import random
import sys, time
import pandas as pd
Ui_MainWindow, QMainWindow = loadUiType('GP_GUI.ui')

class XStream(QObject):
    _stdout = None
    _stderr = None

    messageWritten = pyqtSignal(str)

    def flush( self ):
        pass

    def fileno( self ):
        return -1

    def write( self, msg ):
        if ( not self.signalsBlocked() ):
            self.messageWritten.emit(unicode(msg))

    @staticmethod
    def stdout():
        if ( not XStream._stdout ):
            XStream._stdout = XStream()
            sys.stdout = XStream._stdout
        return XStream._stdout

    @staticmethod
    def stderr():
        if ( not XStream._stderr ):
            XStream._stderr = XStream()
            sys.stderr = XStream._stderr
        return XStream._stderr
class Main(QMainWindow, Ui_MainWindow):


    def __init__(self, parent=None):
        #pyqtgraph.setConfigOption('background', 'w')  # before loading widget
        super(Main, self).__init__()
        self.setupUi(self)
       

        XStream.stdout().messageWritten.connect( self.textBrowser.insertPlainText )
        XStream.stdout().messageWritten.connect( self.textBrowser.ensureCursorVisible )
        XStream.stderr().messageWritten.connect( self.textBrowser.insertPlainText )
        XStream.stderr().messageWritten.connect( self.textBrowser.ensureCursorVisible )
        



        #self.pushButton_20.clicked.connect( self.start_thread0 )
        #self.pushButton_20.setStyleSheet( "background-color: green" ) ## This is to change Button Color



    def start_thread0(self):##Function of any Button
        self.listen.EMG = np.empty( [0, 8] )
        threading.Thread( target=lambda: self.listen.hub.run_forever( self.listen.on_event ) ).start()
        self.flag_thread0 = True
        self.thread0 = threading.Thread(target = self.loop0)
        self.thread0.start()

if __name__ == '__main__':
    import sys
    from PyQt4 import QtGui
    import numpy as np

    app = QtGui.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())



