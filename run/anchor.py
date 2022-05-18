import numpy as np

def get_anchors(config):
    info = config["info"]
    anchors = []
    length = len(info)
    for i in range(length):
        item = info[i]
        cy = np.linspace(item["starts"], item["starts"]+(item["shape"][0]-1)*item["steps"], item["shape"][0])
        cx = np.linspace(item["starts"], item["starts"]+(item["shape"][1]-1)*item["steps"], item["shape"][1])
    
        cx_grid, cy_grid = np.meshgrid(cx, cy)
      
        boxes_tensor = np.zeros((item["shape"][0],item["shape"][1], 8),dtype = np.float32)

        boxes_tensor[:, :, 0] = cx_grid
        boxes_tensor[:, :, 1] = cy_grid
        boxes_tensor[:, :, 2] = item["scale"][0]
        boxes_tensor[:, :, 3] = item["scale"][1]
        boxes_tensor[:, :, 4] = item["boxes"][0]
        boxes_tensor[:, :, 5] = item["boxes"][1]
        boxes_tensor[:, :, 6] = item["ratio"][0]
        boxes_tensor[:, :, 7] = item["ratio"][1]
        boxes_tensor = np.reshape(boxes_tensor,(-1,8))
      #  print(boxes_tensor.shape)
        anchors.append(boxes_tensor)
   # return 
    anchors = np.concatenate(anchors,axis = 0)
    anchors = np.array(anchors,dtype = np.float32)
    return anchors

def get_all_anchors(configs):
    anchors=[]
    for config in configs:
        anchors.append(get_anchors(config))
    return anchors


def iou(left,right):
    if len(left.shape)==1:
        left = np.array([left])
    if len(right.shape)==1:
        right = np.array([right])

    gt_boundary = 0.35

    m,n = left.shape[0],right.shape[0]
    left = np.expand_dims(left,axis = 1)
    left = np.tile(left,(1,n,1))
    right = np.expand_dims(right,axis = 0)
    right = np.tile(right,(m,1,1))
    
    center_x = (right[:,:,0]+right[:,:,2])/2
    diff_x = np.abs((center_x-left[:,:,0])/left[:,:,4])

    center_y = (right[:,:,1]+right[:,:,3])/2
    diff_y = np.abs((center_y-left[:,:,1])/left[:,:,5])

    overlaps = np.sqrt(diff_x**2+diff_y**2)

    diff_x = np.abs((center_x-left[:,:,0])/(right[:,:,2]-right[:,:,0]))
    diff_y = np.abs((center_y-left[:,:,1])/(right[:,:,3]-right[:,:,1]))
    diff = np.sqrt(diff_x**2+diff_y**2)
    
    overlaps[diff>=gt_boundary]=1


    scale = np.maximum(right[:,:,2]-right[:,:,0],right[:,:,3]-right[:,:,1])
    overlaps[scale<=left[:,:,2]]=1
    overlaps[scale>left[:,:,3]]=1

    ratios = (right[:,:,2]-right[:,:,0])/(right[:,:,3]-right[:,:,1])
    overlaps[ratios<left[:,:,6]]=1
    overlaps[ratios>left[:,:,7]]=1


    low = np.minimum(right[:,:,2]-right[:,:,0],right[:,:,3]-right[:,:,1])
    overlaps[low<=6]=1
    return overlaps

if __name__=="__main__":
    config = {"factor":0.5}
    print(get_anchors(config))