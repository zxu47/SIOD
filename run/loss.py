from __future__ import division
import tensorflow as tf
import numpy as np

class SSDLoss80:
  
    def __init__(self,
                 config,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1):

        self.config = config
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha

    def regress_loss(self, y_true, y_pred):

        
        loss = 0
        left = y_true[...,-5:-1]
        nclasses = self.config["nclasses"]
        for i in range(nclasses-1):
            right = y_pred[...,4*i:4*i+4]
            absolute_loss = tf.abs(left - right)
            square_loss = 0.5 * (left - right)**2
            l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
            l1_loss =  tf.reduce_sum(l1_loss, axis=-1)
            state = y_true[...,i+1]
            l1_loss = l1_loss*state

            loss+=tf.reduce_sum(l1_loss)

        localization_loss = loss
        print("localization_loss:",localization_loss)


        state = y_true[:,:,-1]
        positives = tf.where(tf.greater(state,0.5))    
        n_positive = tf.shape(positives)[0]
        loss = localization_loss/tf.maximum(1.0,tf.cast(n_positive,tf.float32))
        print("localization_loss:",loss)
        return self.alpha*loss


    def class_loss(self, y_true, y_pred):

        nclasses = self.config["nclasses"]-1

        batch_size = tf.shape(y_pred)[0] # Output dtype: tf.int32
        n_boxes = tf.shape(y_pred)[1] # Output dtype: tf.int32, note that `n_boxes` in this context denotes the total number of boxes per image, not the number of boxes per cell.

       # classification_loss = tf.cast(self.log_loss(y_true[:,:,:nclasses], y_pred[:,:,:nclasses]),dtype=tf.float32) # Output shape: (batch_size, n_boxes)
      #  localization_loss = tf.cast(self.smooth_L1_loss(y_true, y_pred),dtype=tf.float32) # Output shape: (batch_size, n_boxes)
        
        state = y_true[:,:,-1]
        positives = tf.where(tf.greater(state,0.5))
     #   print("localization_loss:",localization_loss)
        
        
        p1 = y_true[:,:,:(nclasses+1)]
        p2 = y_pred[:,:,:(nclasses+1)]
        classification_loss = -tf.reduce_sum(p1 * tf.math.log(tf.maximum(p2,1e-15)), axis=-1)

        
        n_positive = tf.shape(positives)[0]
        print("n_positives:",n_positive)
        positive_loss = tf.gather_nd(classification_loss,positives)
        positive_loss = tf.reduce_sum(positive_loss)
        print("positive_loss:",positive_loss)
    
       
        negatives = tf.where(tf.equal(state,0))
        negative_loss = tf.gather_nd(classification_loss,negatives)
        # Compute the classification loss for the negative default boxes (if there are any).
        n_negative = tf.shape(negative_loss)[0]

       
        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio*tf.cast(n_positive,tf.int32),self.n_neg_min),tf.cast(n_negative,tf.int32))
        print("n_negative:",n_negative_keep)
        def f1():
            return tf.constant(0.0)
        def f2():
            neg_class_loss_all_1D = tf.reshape(negative_loss,[-1])
            values,indices = tf.nn.top_k(neg_class_loss_all_1D,k = n_negative_keep,sorted = False)

            negatives_keep = tf.scatter_nd(indices = tf.expand_dims(indices,axis = 1),
                                            updates = tf.ones_like(indices,dtype = tf.int32),
                                            shape = tf.shape(neg_class_loss_all_1D))

            negatives_keep = tf.cast(negatives_keep,dtype = tf.float32)
            return tf.reduce_sum(neg_class_loss_all_1D*negatives_keep)

        negative_loss = tf.cond(tf.equal(n_negative,tf.constant(0)),f1,f2)

        print("negative loss:",negative_loss)

        class_loss = tf.reduce_sum(positive_loss)+negative_loss

        total_loss = (class_loss)/tf.maximum(1.0,tf.cast(n_positive,tf.float32))

        print("total loss:",total_loss)
        print("class:",class_loss/tf.maximum(1.0,tf.cast(n_positive,tf.float32)))
       # print("regress:",localization_loss/tf.maximum(1.0,tf.cast(n_positive,tf.float32)))
        return total_loss


class SSDLossSplit:
  
    def __init__(self,
                 config,
                 neg_pos_ratio=3,
                 n_neg_min=10,
                 alpha=1,
                 batch_size = 8):

        self.config = config
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha,
        self.batch_size = batch_size


    def regress_loss(self, y_true, y_pred):

        
        start = self.config["nclasses"]
        left = y_true[...,-5:-1]
        right = y_pred#[...,start:start+4]
        absolute_loss = tf.abs(left-right)
        square_loss = 0.5*(left-right)**2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        l1_loss =  tf.reduce_sum(l1_loss, axis=-1)
      #  l1_loss = l1_loss*self.factor
        temp = tf.reduce_sum(y_true[...,1:start],axis = -1)
        state = tf.where(tf.greater(temp,0.5),tf.ones(shape = tf.shape(y_true[...,-1])),tf.zeros(shape = tf.shape(y_true[...,-1])))
        l1_loss = l1_loss*state

        localization_loss = tf.reduce_sum(l1_loss)
     #   localization_loss = tf.reduce_sum(l1_loss*self.factor)
        print("localization_loss:",localization_loss)


        state = y_true[:,:,-1]
        positives = tf.where(tf.greater(state,0.5))    
        n_positive = tf.shape(positives)[0]
        loss = localization_loss/tf.maximum(1.0,tf.cast(n_positive,tf.float32))
        print("localization_loss:",loss)
        return loss


    def class_loss(self,y_true,y_pred):

        batch_size = tf.shape(y_pred)[0] # Output dtype: tf.int32
      #  n_boxes = tf.shape(y_pred)[1]

        state = y_true[:,:,-1]
        positives = tf.where(tf.greater(state,0.5))
     #   print("localization_loss:",localization_loss)
        
        
        n_positive = tf.shape(positives)[0]

        loss = 0
        for i in range(self.batch_size):
            loss+=self.class_loss_per_batch(y_true[i],y_pred[i])
            print("-------------------------")
        loss=loss/tf.cast(n_positive,tf.float32)
        print("final class loss:",loss)
        return loss

    def class_loss_per_batch(self,y_true, y_pred):

        nclasses = self.config["nclasses"]-1

     #   batch_size = tf.shape(y_pred)[0] # Output dtype: tf.int32
      #  n_boxes = tf.shape(y_pred)[1] # Output dtype: tf.int32, note that `n_boxes` in this context denotes the total number of boxes per image, not the number of boxes per cell.

       # classification_loss = tf.cast(self.log_loss(y_true[:,:,:nclasses], y_pred[:,:,:nclasses]),dtype=tf.float32) # Output shape: (batch_size, n_boxes)
      #  localization_loss = tf.cast(self.smooth_L1_loss(y_true, y_pred),dtype=tf.float32) # Output shape: (batch_size, n_boxes)
        
        state = y_true[:,-1]
        positives = tf.where(tf.greater(state,0.5))
     #   print("localization_loss:",localization_loss)
        
        
        p1 = y_true[:,:(nclasses+1)]
        p2 = y_pred[:,:(nclasses+1)]
        classification_loss = -tf.reduce_sum(p1 * tf.math.log(tf.maximum(p2,1e-15)), axis=-1)
      #  classification_loss = classification_loss*self.factor

        
        n_positive = tf.shape(positives)[0]
        print("n_positives:",n_positive)
        positive_loss = tf.gather_nd(classification_loss,positives)
        positive_loss = tf.reduce_sum(positive_loss)
        print("positive_loss:",positive_loss)
    
       
        negatives = tf.where(tf.equal(state,0))
        negative_loss = tf.gather_nd(classification_loss,negatives)
        # Compute the classification loss for the negative default boxes (if there are any).
        n_negative = tf.shape(negative_loss)[0]

       
        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio*tf.cast(n_positive,tf.int32),self.n_neg_min),tf.cast(n_negative,tf.int32))
        print("n_negative:",n_negative_keep)
        def f1():
            return tf.constant(0.0)
        def f2():
            neg_class_loss_all_1D = tf.reshape(negative_loss,[-1])
            values,indices = tf.nn.top_k(neg_class_loss_all_1D,k = n_negative_keep,sorted = False)

            negatives_keep = tf.scatter_nd(indices = tf.expand_dims(indices,axis = 1),
                                            updates = tf.ones_like(indices,dtype = tf.int32),
                                            shape = tf.shape(neg_class_loss_all_1D))

            negatives_keep = tf.cast(negatives_keep,dtype = tf.float32)
            return tf.reduce_sum(neg_class_loss_all_1D*negatives_keep)

        negative_loss = tf.cond(tf.equal(n_negative,tf.constant(0)),f1,f2)

        print("negative loss:",negative_loss)

        class_loss = tf.reduce_sum(positive_loss)+negative_loss

      #  class_loss = (class_loss)/tf.maximum(1.0,tf.cast(n_positive,tf.float32))

      #  print("total loss:",total_loss)
        print("class:",class_loss)
       # print("regress:",localization_loss/tf.maximum(1.0,tf.cast(n_positive,tf.float32)))
        return class_loss



class SSDLoss:
  
    def __init__(self,
                 config,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1):

        self.config = config
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha

    def regress_loss(self, y_true, y_pred):

        
        start = self.config["nclasses"]
        left = y_true[...,-5:-1]
        right = y_pred#[...,start:start+4]
        absolute_loss = tf.abs(left-right)
        square_loss = 0.5*(left-right)**2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        l1_loss =  tf.reduce_sum(l1_loss, axis=-1)
      #  l1_loss = l1_loss*self.factor
        temp = tf.reduce_sum(y_true[...,1:start],axis = -1)
        state = tf.where(tf.greater(temp,0.5),tf.ones(shape = tf.shape(y_true[...,-1])),tf.zeros(shape = tf.shape(y_true[...,-1])))
        l1_loss = l1_loss*state

        localization_loss = tf.reduce_sum(l1_loss)
     #   localization_loss = tf.reduce_sum(l1_loss*self.factor)
        print("localization_loss:",localization_loss)


        state = y_true[:,:,-1]
        positives = tf.where(tf.greater(state,0.5))    
        n_positive = tf.shape(positives)[0]
        loss = localization_loss/tf.maximum(1.0,tf.cast(n_positive,tf.float32))
        print("localization_loss:",loss)
        return loss


    def class_loss(self, y_true, y_pred):

        nclasses = self.config["nclasses"]-1

        batch_size = tf.shape(y_pred)[0] # Output dtype: tf.int32
        n_boxes = tf.shape(y_pred)[1] # Output dtype: tf.int32, note that `n_boxes` in this context denotes the total number of boxes per image, not the number of boxes per cell.

       # classification_loss = tf.cast(self.log_loss(y_true[:,:,:nclasses], y_pred[:,:,:nclasses]),dtype=tf.float32) # Output shape: (batch_size, n_boxes)
      #  localization_loss = tf.cast(self.smooth_L1_loss(y_true, y_pred),dtype=tf.float32) # Output shape: (batch_size, n_boxes)
        
        state = y_true[:,:,-1]
        positives = tf.where(tf.greater(state,0.5))
     #   print("localization_loss:",localization_loss)
        
        
        p1 = y_true[:,:,:(nclasses+1)]
        p2 = y_pred[:,:,:(nclasses+1)]
        classification_loss = -tf.reduce_sum(p1 * tf.math.log(tf.maximum(p2,1e-15)), axis=-1)
      #  classification_loss = classification_loss*self.factor

        
        n_positive = tf.shape(positives)[0]
        print("n_positives:",n_positive)
        positive_loss = tf.gather_nd(classification_loss,positives)
        positive_loss = tf.reduce_sum(positive_loss)
        print("positive_loss:",positive_loss)
    
       
        negatives = tf.where(tf.equal(state,0))
        negative_loss = tf.gather_nd(classification_loss,negatives)
        # Compute the classification loss for the negative default boxes (if there are any).
        n_negative = tf.shape(negative_loss)[0]

       # n_negative_keep = tf.minimum(tf.maximum(tf.cast(tf.cast(n_positive,tf.float32)*tf.cast(self.neg_pos_ratio,tf.float32),tf.int32),self.n_neg_min),tf.cast(n_negative,tf.int32))       
        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio*tf.cast(n_positive,tf.int32),self.n_neg_min),tf.cast(n_negative,tf.int32))
        print("n_negative:",n_negative_keep)
        def f1():
            return tf.constant(0.0)
        def f2():
            neg_class_loss_all_1D = tf.reshape(negative_loss,[-1])
            values,indices = tf.nn.top_k(neg_class_loss_all_1D,k = n_negative_keep,sorted = False)

            negatives_keep = tf.scatter_nd(indices = tf.expand_dims(indices,axis = 1),
                                            updates = tf.ones_like(indices,dtype = tf.int32),
                                            shape = tf.shape(neg_class_loss_all_1D))

            negatives_keep = tf.cast(negatives_keep,dtype = tf.float32)
            return tf.reduce_sum(neg_class_loss_all_1D*negatives_keep)

        negative_loss = tf.cond(tf.equal(n_negative,tf.constant(0)),f1,f2)

        print("negative loss:",negative_loss)

        class_loss = tf.reduce_sum(positive_loss)+negative_loss

        total_loss = (class_loss)/tf.maximum(1.0,tf.cast(n_positive,tf.float32))

        print("total loss:",total_loss)
        print("class:",class_loss/tf.maximum(1.0,tf.cast(n_positive,tf.float32)))
       # print("regress:",localization_loss/tf.maximum(1.0,tf.cast(n_positive,tf.float32)))
        return total_loss



class SSDLossTest:
  
    def __init__(self,
                 config,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1):

        self.config = config
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha


    def regress(self,y_true, y_pred):
        start = 0
        info = self.config["info"]
        length = len(info)
        for i in range(length):
            item = info[i]
            a,b = item["shape"]
            end = start+a*b
            print("------------------\n i = "+str(i))
            self.regress_loss(y_true[:,start:end,:],y_pred[:,start:end,:])
            start = end


    def classifier(self,y_true, y_pred):
        start = 0
        info = self.config["info"]
        length = len(info)
        for i in range(length):
            item = info[i]
            a,b = item["shape"]
            end = start+a*b
            print("------------------\n i = "+str(i))
            self.class_loss(y_true[:,start:end,:],y_pred[:,start:end,:])
            start = end

    def regress_loss(self, y_true, y_pred):

        
        start = self.config["nclasses"]
        left = y_true[...,-5:-1]
        right = y_pred#[...,start:start+4]
        absolute_loss = tf.abs(left-right)
        square_loss = 0.5*(left-right)**2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        l1_loss =  tf.reduce_sum(l1_loss, axis=-1)
        temp = tf.reduce_sum(y_true[...,1:start],axis = -1)
        state = tf.where(tf.greater(temp,0.5),tf.ones(shape = tf.shape(y_true[...,-1])),tf.zeros(shape = tf.shape(y_true[...,-1])))
        l1_loss = l1_loss*state

        localization_loss = tf.reduce_sum(l1_loss)
        print("localization_loss:",localization_loss)


        state = y_true[:,:,-1]
        positives = tf.where(tf.greater(state,0.5))    
        n_positive = tf.shape(positives)[0]
        loss = localization_loss/tf.maximum(1.0,tf.cast(n_positive,tf.float32))
        print("localization_loss:",loss)
        return loss


    def class_loss(self, y_true, y_pred):

        nclasses = self.config["nclasses"]-1

        batch_size = tf.shape(y_pred)[0] # Output dtype: tf.int32
        n_boxes = tf.shape(y_pred)[1] # Output dtype: tf.int32, note that `n_boxes` in this context denotes the total number of boxes per image, not the number of boxes per cell.

       # classification_loss = tf.cast(self.log_loss(y_true[:,:,:nclasses], y_pred[:,:,:nclasses]),dtype=tf.float32) # Output shape: (batch_size, n_boxes)
      #  localization_loss = tf.cast(self.smooth_L1_loss(y_true, y_pred),dtype=tf.float32) # Output shape: (batch_size, n_boxes)
        
        state = y_true[:,:,-1]
        positives = tf.where(tf.greater(state,0.5))
     #   print("localization_loss:",localization_loss)
        
        
        p1 = y_true[:,:,:(nclasses+1)]
        p2 = y_pred[:,:,:(nclasses+1)]
        classification_loss = -tf.reduce_sum(p1 * tf.math.log(tf.maximum(p2,1e-15)), axis=-1)

        
        n_positive = tf.shape(positives)[0]
        print("n_positives:",n_positive)
        positive_loss = tf.gather_nd(classification_loss,positives)
        positive_loss = tf.reduce_sum(positive_loss)
        print("positive_loss:",positive_loss)
    
       
        negatives = tf.where(tf.equal(state,0))
        negative_loss = tf.gather_nd(classification_loss,negatives)
        # Compute the classification loss for the negative default boxes (if there are any).
        n_negative = tf.shape(negative_loss)[0]

       
        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio*tf.cast(n_positive,tf.int32),self.n_neg_min),tf.cast(n_negative,tf.int32))
        print("n_negative:",n_negative_keep)
        def f1():
            return tf.constant(0.0)
        def f2():
            neg_class_loss_all_1D = tf.reshape(negative_loss,[-1])
            values,indices = tf.nn.top_k(neg_class_loss_all_1D,k = n_negative_keep,sorted = False)

            negatives_keep = tf.scatter_nd(indices = tf.expand_dims(indices,axis = 1),
                                            updates = tf.ones_like(indices,dtype = tf.int32),
                                            shape = tf.shape(neg_class_loss_all_1D))

            negatives_keep = tf.cast(negatives_keep,dtype = tf.float32)
            return tf.reduce_sum(neg_class_loss_all_1D*negatives_keep)

        negative_loss = tf.cond(tf.equal(n_negative,tf.constant(0)),f1,f2)

        print("negative loss:",negative_loss)

        class_loss = tf.reduce_sum(positive_loss)+negative_loss

        total_loss = (class_loss)/tf.maximum(1.0,tf.cast(n_positive,tf.float32))

        print("total loss:",total_loss)
        print("class:",class_loss/tf.maximum(1.0,tf.cast(n_positive,tf.float32)))
       # print("regress:",localization_loss/tf.maximum(1.0,tf.cast(n_positive,tf.float32)))
        return total_loss
