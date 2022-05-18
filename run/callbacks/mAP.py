import tensorflow as tf



class mAPCallbackNew(tf.keras.callbacks.Callback):

    def __init__(self,config,worker,index = 1):
       
        self.config = config
        self.worker = worker
        self.index = index
      #  self.tensor_board = tensor_board
        super(mAPCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if epoch%self.index!=0:
            logs['mAP']=0
            for i in range(1,21):
               logs['class_'+str(int(i/10))+str(i%10)]=0 
            return

        mAP,details = self.worker.process()

        logs['mAP'] = mAP
        for i in range(1,21):
            logs['class_'+str(int(i/10))+str(i%10)]=details[i]

      #  print('mAP: {:.4f}'.format(mAP))


class mAPCallbackNew1(tf.keras.callbacks.Callback):

    def __init__(self,config,worker,index = 1):
       
        self.config = config
        self.worker = worker
        self.index = index
      #  self.tensor_board = tensor_board
        super(mAPCallbackNew1, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if epoch%self.index!=0:
            logs['mAP']=0
            for i in range(1,21):
               logs['class_'+str(int(i/10))+str(i%10)]=0 
            return

        mAP,details = self.worker.process()

        logs['mAP'] = mAP
        for i in range(1,21):
            logs['class_'+str(int(i/10))+str(i%10)]=details[i]

      #  print('mAP: {:.4f}'.format(mAP))
