import numpy as np 

### All Required variabel initialized################## 
image_x,image_y=28,28
I_len=576
Kernal_size_L1=3
No_Kernal_L1=8
Kernal_size_L2=3
No_Kernal_L2=16
O_node=10  
lr=0.005
########################################################

###RAndom Initializer###################################
weights=np.random.randn(I_len,O_node)/I_len
biases=np.random.randn(1,O_node)
Kernals_L1=np.random.randn(No_Kernal_L1,Kernal_size_L1,Kernal_size_L1)/(Kernal_size_L1**2)
Kernals_L2=np.random.randn(No_Kernal_L2,Kernal_size_L2,Kernal_size_L2)/(Kernal_size_L2**2)
###########################################################

###All Interlinked variable declared########################
Weighted_Sum,baises,input_nn=[],[],[]
Softmax,Pooling_Output,Pooling_Output_L2=[],[],[]
Conv_Output,Conv_Output_L2=[],[]
image,image_M,Actual_Label=[],[],[]
############################################################  
  
  
def main():
    global image,Actual_Label,Softmax
    inc=0
    total=60
    
    for i in range(1000):
        image=t_d[i][0].reshape(28,28)-0.5
        Actual_Label=np.argmax(t_d[i][1])
        train()
        
       
           
    for i in range(total):
        image=t_d[45000+i][0].reshape(28,28)-0.5
        Actual_Label=np.argmax(t_d[45000+i][1])
        test()
        if np.argmax(Softmax)==Actual_Label:
            inc+=1
     
    percent=(inc/total)*100
    print("Accuracy:",percent,"%")


def test():
    
    global Weighted_Sum,weights,biases,Kernals_L1,Actual_Label,Conv_Output
    global Softmax,input_nn,lr,image_M,I_len,Pooling_Output,Conv_Output_L2
    global Pooling_Output_L2,Kernals_L2
    global image_x,image_y
    
    ##Layer1############################################################
    #Padding
    image_M=np.zeros((image_x+2,image_y+2))
    image_M[1:(1+image_x),1:(1+image_y)]=image_M[1:(1+image_x),1:(1+image_y)]+image
    
    
    #Convolution Part
    Conv_Output=np.zeros((No_Kernal_L1,image_x,image_y)) #8x28x28
    for k in range(No_Kernal_L1):
        for i in range(image_x):
            for j in range(image_y):
                img_section=image_M[i:(i+Kernal_size_L1),j:(j+Kernal_size_L1)]
                Conv_Output[k,i,j]=(Kernals_L1[k]*img_section).sum()       
    
  
    #Relu
    for k in range(No_Kernal_L1):    
        for i in range(image_x):
            for j in range(image_y):
                if Conv_Output[k][i][j] < 0:            
                    Conv_Output[k][i][j]=0
                    
            
    #Pooling
    h,w=image_x//2,image_y//2
    Pooling_Output=np.zeros((No_Kernal_L1,h,w))  #8x14x14            
    for k in range(No_Kernal_L1):    
        for i in range(h):
            for j in range(w):
                img_section=Conv_Output[k][(i*2):(i*2+2),(j*2):(j*2+2)]
                Pooling_Output[k,i,j]=np.amax(img_section)
###################################################################            

####Layer2#####################################################
    #Convolution Part
    Conv_Output_L2=np.zeros((No_Kernal_L2,h-2,w-2)) #8x12x12
    for k in range(No_Kernal_L2):
        for i in range(h-2):
            for j in range(w-2):
                img_section=Pooling_Output[k][i:(i+Kernal_size_L2),j:(j+Kernal_size_L2)]
                Conv_Output_L2[k,i,j]=(Kernals_L2[k]*img_section).sum()   #Conv_output -8 x 3 x 3


    #Relu
    for k in range(No_Kernal_L2):    
        for i in range(h-2):
            for j in range(w-2):
                if Conv_Output_L2[k][i][j] < 0:            
                    Conv_Output_L2[k][i][j]=0
                    
            
    #Pooling
    h,w=(h-2)//2,(w-2)//2
    Pooling_Output_L2=np.zeros((No_Kernal_L2,h,w)) #8x6x6            
    for k in range(No_Kernal_L2):    
        for i in range(h):
            for j in range(w):
                img_section=Conv_Output_L2[k][(i*2):(i*2+2),(j*2):(j*2+2)]
                Pooling_Output_L2[k,i,j]=np.amax(img_section)
#################################################################################
                
    #x,y,z=Pooling_Output.shape
    #I_len=x*y*z
    
    #Softmax/ANN part
    input_nn=Pooling_Output_L2.flatten().reshape(1,I_len)
    
    Weighted_Sum=np.dot(input_nn,weights)+biases
   
    exp=np.exp(Weighted_Sum)
    exp2=exp.sum()
    
    Softmax=exp/exp2
    
    
    # Cross entropy loss
    loss=-np.log(Softmax[0][Actual_Label])
    
    

    
def train():
    global Weighted_Sum,weights,biases,Kernals_L1,Actual_Label,Conv_Output,Conv_Output_L2
    global Softmax,input_nn,lr,Pooling_Output,Pooling_Output_L2,Kernals_L2,image_M
    global image_x,image_y
    
    test()
    
    Output_Gradient=np.zeros((1,O_node))
    Output_Gradient[0][Actual_Label]=-1/Softmax[0][Actual_Label]
    
    for i in range(O_node):
        if Output_Gradient[0][i]==0:
            continue
        Weighted_Sum_exp=np.exp(Weighted_Sum)
        
        S=Weighted_Sum_exp.sum()
        
        #Gradient of Softmax w.r.t weighted sum
        de_Soft__de_WS=-Weighted_Sum_exp[0][i]* Weighted_Sum_exp/(S**2)
        
        de_Soft__de_WS[0][i]=Weighted_Sum_exp[0][i]*(S- Weighted_Sum_exp[0][i])/(S**2)
        
        
        #Gradient of weightedsum w.r.t weights
        de_WS__de_W=input_nn
        de_WS__de_Bias=1
        de_WS__de_Input=weights
        
        #Gradient of Loss w.r.t Weighted sum
        de_Loss__de_WS=Output_Gradient[0][i]* de_Soft__de_WS
        
        #Gradient of Loss w.r.t weights/biases/input
        de_Loss__de_W=np.dot(de_WS__de_W.reshape(I_len,1),de_Loss__de_WS)
        de_Loss__Bias=de_Loss__de_WS*de_WS__de_Bias
        x,y,z=Pooling_Output_L2.shape
        de_Loss__de_Input=(np.dot(de_WS__de_Input,de_Loss__de_WS.reshape(O_node,1)).reshape(1,I_len)).reshape(No_Kernal_L2,y,z)
        #if i%100==0:
        weights -=lr*de_Loss__de_W
        biases -=lr*de_Loss__Bias
        
        
#############################################################################
####Layer 2################################################
    #Pooling Backprop
    x,y,z=Conv_Output_L2.shape
    de_pool_L2=np.zeros((No_Kernal_L2,y,z)) #8x12x12
        
    for k in range(No_Kernal_L2):
        for i in range(y//2): #6x6
            for j in range(z//2):
                img_section=np.argmax(Conv_Output_L2[k][2*i:(2*i+2),2*j:(2*j+2)])
                #index value
                qnt=img_section//2
                rm=img_section%2
                de_pool_L2[k][qnt+2*i,rm+2*j]=de_Loss__de_Input[k][i][j]                    
                    
    #Gradient of kernel wrt pool
    de_Loss_de_filter_L2=np.zeros((No_Kernal_L2,Kernal_size_L2,Kernal_size_L2)) #8x3x3
    for k in range(No_Kernal_L2):
        for i in range(y): #12x12
            for j in range(z):
                img_section=Pooling_Output[k][i:(i+Kernal_size_L2),j:(j+Kernal_size_L2)]
                de_Loss_de_filter_L2[k] +=de_pool_L2[k][i,j]*img_section
                

    #Gradient of convolution input de_pool_L2 as output gradient
    O_x,O_y=y,z  #12x12
    O=np.zeros((No_Kernal_L2,y,z))
    x,y,z=Pooling_Output.shape
    de_Conv=np.zeros((No_Kernal_L2,y,z))
    m,n=0,0
    
    for k in range(No_Kernal_L2):
        for i in range(O_x):   #Output gradient x-axis
            for j in range(O_y):
                O[k][i,j]=de_pool_L2[k][i,j]
                for p in range(i,i+Kernal_size_L2):
                    for q in range(j,j+Kernal_size_L2):
                        de_Conv[k][p,q]=O[k][i,j]*Kernals_L2[k][m,n]+de_Conv[k][p,q]
                        n+=1
                    m+=1
                    n=0
                m=0
    #Update Kernals
    #if i%100==0:
    Kernals_L2 -= lr*de_Loss_de_filter_L2

################################################################################
    ##Layer 1
    #Pooling Backprop
    de_pool=np.zeros((No_Kernal_L1,image_x,image_y))
    x,y,z=Pooling_Output.shape
    for k in range(No_Kernal_L1):
        for i in range(y):
            for j in range(z):
                img_section=np.argmax(Conv_Output[k][2*i:(2*i+2),2*j:(2*j+2)])
                #index value111
                qnt=img_section//2
                rm=img_section%2
                #if i%2==0 and j%2==0:    
                de_pool[k][qnt+2*i,rm+2*j]=de_Conv[k][i][j]                    

    #Gradient of kernel wrt pool
    de_Loss_de_filter=np.zeros((No_Kernal_L1,Kernal_size_L1,Kernal_size_L1))
    for k in range(No_Kernal_L1):
        for i in range(image_x):
            for j in range(image_y):
                img_section=image_M[i:(i+Kernal_size_L1),j:(j+Kernal_size_L1)]
                de_Loss_de_filter[k] +=de_pool[k][i,j]*img_section

                
    #Update Kernals
    #if i%100==0:
    Kernals_L1 -= lr*de_Loss_de_filter
                

main()

