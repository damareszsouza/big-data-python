# big-data-python
Exemplo curso big data IBM


Research Computing at Stern/NYU This blog will be used to share information about research computing in business schools. Hopefully other researchers around the world will provide some feedback. I hope it is helpful. Norman White, Faculty Director, Stern Center for Research Computing
procurar

NOV
19
Stern HPC moving to Slurm for scheduling
As part of our move to better support deep learning, we have decided to migrate from the grid processing scheduler we currently use (Oracle Grid Engine) (no longer supported) to the SLURM Workload Manager.

 Slurm is the most popular workload manager at large HPC clusters, and has sophisticated support for GPU nodes. In addition, it is the scheduler used at the much larger NYU HPC sites, so our users can easily transition to that environment. We hope to be able to move to Slurm during the next semester as we slowly decommission our Grid Engine nodes. This should be straightforward, since many of our nodes run in our VMWARE private research cloud. We will just reduce the number of grid engine VMs as we increase the number of Slurm nodes.

Publicado em 19 de novembro de 2019 por Norman White
Marcadores: gpu  Grid Engine  HPC  nuvem privada  SLURM  Stern Center for Research Computing  vmware
 
2 Ver comentários
NOVEMBRO
19
Mais GPUs na grade Stern
O cluster Stern hpc adicionou recentemente um novo servidor, com 8 GPUs NVIDIA 2780ti. Essas GPUs não são de alto desempenho como o NIDIA V100, mas são muito mais baratas. Tensorflow, Keras etc, podem usar várias GPUs em paralelo, então isso deve ser perfeito para aprendizado profundo.
Cada 2780 ti é cerca de 80% mais rápido que um V100 e tem 11 GB de RAM (em oposição aos 16 GB do V100). O servidor em que estão tem 392GB de RAM e um processador muito rápido com instruções AVX 512. Esperamos ver em breve algumas pesquisas pesadas sobre aprendizado profundo.
Publicado em 19 de novembro de 2019 por Norman White
Marcadores: aprendizado profundo  HPC  keras  nvidia 2780  nvidia V100  nyu  computação de pesquisa  Stern School of Business  tensorflow
 
1 Ver comentários
MARÇO
12
Usando o Tensorflow no Stern HPC Grid
Tensorflow na grade Stern
Usar o tensorflow com uma gpu na grade Stern envolve uma configuração única de um ambiente virtual para seu código python. O ambiente virtual permitirá que você instale qualquer módulo python que desejar, incluindo o tensorflow. No entanto, para rodar na grade Stern, você deve instalar tensorflow-gpu==1.13.1.
Aqui estão as etapas a serem executadas no rnd para configurar inicialmente seu ambiente. Isso pressupõe o uso do shell bash. (OBSERVE USUÁRIOS DO PYTORCH, a mesma abordagem também funcionará para o pytorch)

PRIMEIRA CONFIGURAÇÃO

A) Crie um diretório para armazenar seu ambiente python. Aqui assumimos que você está chamando esse diretório de "mytensorwork"

Emita os seguintes comandos (uma vez em rnd) para criar inicialmente seu ambiente virtual python.
cd
 mkdir ~/mytensorwork
virtualenv -p /gridapps/Python-3.6.5/bin/python mytensorwork
B) ative sua versão privada do python (você precisa fazer isso toda vez que executar). 
cd mytensorwork
fonte bin/ativar
C) Instale o Tensorflow (e quaisquer outros módulos python necessários)
pip install tensorflow-gpu==1.13.1 (ou 1.13.1rc2 pode ser uma atualização)
pip instalar...
 EXECUTAR seu código tensorflow na grade Stern em uma GPU
Agora você precisa criar um trabalho para executar seu aplicativo tensorflow. Vamos supor que é um arquivo .py chamado mytensor.py e está em seu diretório mytensorwork.

Seu arquivo shell a ser executado ficará assim:
 mytensorjob.sh
#$ -l GPU=1
#$ -N meutensorjob
#$ -l h_vmem=24G
#$ -l h_cpu =1:00:00
cd mytensorwork
fonte bin/ativar
python mytensor.py


Como teste, você pode copiar o arquivo mnist.py de /gridapps/tfexamples para seu
diretório local e executá-lo. Ele pega um conjunto rotulado de imagens de dígitos e os classifica.

O Tensorflow deve encontrar automaticamente a gpu e usá-la. Se você deseja ver qual é a aceleração da gpu, você deve ser capaz de executar o código em um nó não gpu e ver o quão rápido ele é executado sem a gpu. Mas você precisa fazer isso em um ambiente virtual que NÃO tenha a versão gpu do tensorflow. (ou seja, pip install tensorflow==1.13.1 )

Executando o código Keras 
Você também pode adicionar o código keras , apenas
pip instalar keras 

Depois de criar o ambiente virtual tensorflow e instalar o tensorflow-gpu.

Exemplos de Tensorflow e keras
Você pode encontrar exemplos de tensorflow e keras em /gridapps/tfexamples.

Eles devem ser executados sem alterações.
Por exemplo,
python /gridapps/tfexamples/mnist.py
Executará o código abaixo:



  """ Rede neural convolucional.                                                                                             
Construa e treine uma rede neural convolucional com o TensorFlow.                                                               
Este exemplo está usando o banco de dados MNIST de dígitos manuscritos                                                                
(http://yann.lecun.com/exdb/mnist/)                                                                                           
Autor: Aymeric Damien                                                                                                        
Project: https ://github.com/aymericdamien/TensorFlow-Examples/                                                                
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf

# PRIMEIRO: VEJA SE A GPU ESTÁ SENDO USADA!!!                                                                                          


a = tf.constant([1,0, 2,0, 3,0, 4,0, 5,0, 6,0], forma=[2, 3], nome='a')
b = tf.constant([1,0, 2,0, 3,0, 4,0, 5,0 , 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print (sess.run (c))


# Importar dados MNIST                                                                                                           
de tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
  # Parâmetros de treinamento                                                                                                         
learning_rate = 0,001
num_steps = 1000
batch_size = 128
display_step = 10

# Parâmetros de rede                                                                                                          
num_input = 784 # Entrada de dados MNIST (formato img: 28*28)                                                                         
num_classes = 10 # total de classes MNIST (0-9 dígitos)                                                                           
dropout = 0,75 # Desistência, probabilidade de manter unidades                                                                           

# tf Graph input                                                                                                              
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probabilidade)                                                           


# Crie alguns wrappers para simplificar                                                                                         
def conv2d(x, W, b, strides=1):
    # Wrapper Conv2D, com ativação de bias e                                                                           
    relu x = tf.nn.conv2d(x, W, strides=[1, strides, passadas, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper                                                                                                        
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME ')


# Cria o modelo                                                                                                                 
def conv_net(x, weights, biases, dropout):
    # A entrada de dados MNIST é um vetor 1-D de 784 feições (28*28 pixels)                                                          
    # Remodela para corresponder ao formato da imagem [Altura x Largura x Canal]                                                               
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]                                                            
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer                                                                                                        
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)                                                                                              
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer                                                                                                        
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)                                                                                              
    conv2 = maxpool2d(conv2, k=2)
    # Fully connected layer                                                                                                    
    # Reshape conv2 output to fit fully connected layer input                                                                  
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout                                                                                                            
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction                                                                                                 
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias                                                                                                   
weights = {
    # 5x5 conv, 1 input, 32 outputs                                                                                            
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs                                                                                          
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs                                                                             
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)                                                                               
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model                                                                                                              
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer                                                                                                    
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model                                                                                                               
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)                                                                   
init = tf.global_variables_initializer()

# Start training                                                                                                               
with tf.Session() as sess:
    # Run the initializer                                                                                                      
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)                                                                                       
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
    if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy                                                                                
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images                                                                             
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y: mnist.test.labels[:256],
keep_prob: 1.0}))


Posted 12th March 2019 by Norman White
Labels: deep learning gpu grid computing HPC keras nyu python stern tensorflow
 
1 View comments

MAR
6
Using GPUs with Matlab on the Stern HPC cluster
Using Matlab and GPUs at Stern
Using Matlab and the new NVidia Volta V100 on the Stern HPC cluster is straightforward.

1) First you need to modify your matlab code (slightly) 

You need to  indicate the data elements you want to move to the gpu by using the gpuArray function. You only want to use the gpu for large arrays due to the overhead of moving the data to the gpu and then retrieving it. Once a variable is in the gpu, matlab will automatically use the gpu for any computations involving that variable. The best uses are when you have a number of computations that can be done in the GPU, since there is significant overhead in copying data into  the GPU and retrieving results, as you will see from the examples.

Here is a simple example of using the fft function in matlab:
Sample code not using the GPU:
A1 = rand(3000,3000);
tic;
B1 =fft(A1)
time1 = toc;
 
Sample code using the GPU:  (B2 is left on GPU)
% convert A1 into an array(A2) on the gpu
% computations involving A2 will be done on the gpu (including results)
A2 = gpuArray(A1)
tic;
B2=fft(A2)
time2= toc;
% Speedup of GPU
speedup = time1/time2;
disp(speedup)
 
2)  You need to submit your code to run on the GPU node
To submit your code to run on the GPU node on the Stern grid, you need to add :
#$  -l  GPU=1
to your grid submit file:
Here is an example of a file to submit for processing;
#!/bin/sh                                                                 
#$ -l h_cpu=0:5:00                                                    
#$ -l h_vmem=32G                                                          
#$ -l GPU=1                                                               
module load matlab/2018b
which matlab
echo $PATH
matlab -nojvm


You can get the code for these and other examples at:

 http://people.stern.nyu.edu/nwhite/matlabgpu



You can find more detailed examples on the mathworks web site.
 https://blogs.mathworks.com/loren/2012/02/06/using-gpus-in-matlab/

 
 
 
 
 
 
Posted 6th March 2019 by Norman White
 
1 View comments
MAR
6
GPU Computing comes to Stern
 Background
The recent dramatic advances of "Deep Learning" and it's many applications have made it a standard tool in data science, often out-performing other data mining techniques. But this performance comes at a cost. The multi-layered (deep) neural nets used in deep learning are extremely computationally intensive and often infeasible to train using standard desktop machines, or even powerful multi-core servers. To solve this problem, manufacturers like nvidia, amd, intel, ... have adapted their graphics processing units (GPUs) from just doing some of the low level operations need to scale, rotate, and move graphics objects to include higher level operations need by many types of engineering, statistical, physical, biological , etc modeling. These include vector and tensor analysis, matrix inversion and  other type of operations used by systems like matlab, R, tensorflow, stata, numpy, sci-kit learn  and many other matrix based systems. If applications used the GPUs for calculations, they could run 10 to 100 times faster (and sometimes more). At first, users had to write low level C code to access the capabilities of the GPUs, but GPU manufacturers quickly developed software libraries that (mostly) hid the intricacies of the GPU from programmers. Advanced analytical software like matlab, numpy, and many other applications started to support using GPUs with very few program changes as they incorporated the libraries of the GPU manufacturers into their products. Bitcoin miners also started using GPUs to speed their computations. These applications,  in turn,  encouraged the GPU manufacturers to start developing GPUs which were not designed for graphics, but rather just for computation.


Deep Learning and GPUs

At about the same time, some researchers were experimenting with a new neural net architecture, which involved multiple (sometimes many) levels of networks. The name "Deep Learning" was applied to these new approaches. These networks would take a very long time to converge, but often had superior results on many of the standard machine learning data sets , like number recognition and image recognition. As researchers started to use GPUs, they could use networks which had been computationally prohibitive without GPUs. Fast forward to today, and these deep learning networks are being used by companies like Google, Apple, Microsoft , Facebook, ... in their systems, and we interact with them every day. Soon they may drive our cars for us. However, the application of these systems is just starting, with research being done across many fields, including medicine, busing, government, military, ... In order to enable Stern researchers to access these capabilities easily, research computing has just installed a GPU server.

Stern GPU capabilities

The new server is a Dell R740 with 48 cores, and 392GB of ram. It also has an Nvidia V100 GPU. The GPU has 16GB of ram, and the equivalent of about 5000 cores of processing. 

The GPU is supported by matlab, tensorflow, pytorch and most other data science packages. We have just completed tests using matlab that indicate it is VERY fast on standard benchmarks (about 50% faster than the previous K80 Nvidia GPUs). The server will be available on the Stern HPC grid within a few days, and can be selected by requesting the attribute GPU = 1. Instructions and examples on how to use it  for different applications will be made available in separate posts.


Posted 6th March 2019 by Norman White
Labels: amd Dell gpu HPC matlab Nvidia nvidia V100 nvidia volta tensor tensor flow
 
1 View comments
DEC
20
Catching up - Kubernetes et al...
 Catching up
It has been more than a year since my last post, as I dealt with a busy year and some major health problems. As I look back, I see that the world has had some major changes in research computing. There have been some recent changes at NYU/Stern as well.
New Servers : Knights Landing
Last year, Dell donated/lent two Intel Xeon Phi Knights Landing servers to experiment with. These servers are an attempt to avoid having to use GPUs for very compute intensive tasks that can be parallelized (think deep learning). Their advantage is that there are virtually no code changes necessary. The disadvantages are:
1) The application must be able to use many, many threads of processing. (Each server can run 256 processing threads in parallel).
2) The single core processing speed is slow (presumably to allow so many threads without overheating).
3) There is no virtualization support.

Although a promising approach which has been used in some of the fastest super computers in the world, we still haven't found them as useful in our environment as faster GPU equipped servers.
Intel seems to agree, having recently dropped the approach . We will use our servers specifically for running Deep Learning applications, for which it seems very suitable.
Openstack, Kubernetes and Docker Containers for Jupyterhub
Meanwhile, the virtualization scene is changing rapidly, with new entries like  Openstack and kubernetes based containers. These two approaches are starting to put  significant headwinds in front of VMware (and Dell), especially now that IBM has bought Red Hat.

NYU has a pilot Openstack cluster which has recently been extended to run kubernetes and docker containers. This architecture is being used to deploy  a kubernetes based jupyterhub environment used for classroom instruction of a number of Data Science classes at the Stern School of Business.
This allows the class to have their own jupyter/ipython notebook running in a docker container. Kubernetes is used to automatically scale the approach to as many physical servers are as necessary, by automatically finding a server to run a container on.

A kubernetes cluster can be used to build a cloud of docker containers that can easily be expanded, as well as supporting many different applications (think of HPC jobs running in docker containers, hadoop nodes being automatically added when necessary).
The Convergence of instructional and research computing
The one major difference in the last several years has been the rapid increase in the use of research computing applications in classroom teaching. This presents  a set of new organizational challenges tot he support staff, who suddenly are responsible for many thousands of students instead of hundreds of researchers. Instructional computing has much higher demands on uptime, performance, documentation, support, ... than research computing. I will talk more about this in a future post...

Posted 20th December 2018 by Norman White
Labels: containers Dell docker IBM ipython jupyter jupyterhub kubernetes openstack Red Hat research computing vmware
 
0 Add a comment
NOV
28
New Hardware at the Center for Research Computing
We recently installed the next generation of hardware at the Center. It is a Dell FX-2 "converged" chassis, with a Dell FC630 server. In addition, we have added a fourth Netgear 10gb switch (XS728T), as we transition to a 10gb backbone network within the center.

The Dell Fx2 Chassis is a 2u chassis that has 4 locations that can take either FC630 servers or storage. The chassis comes with dual power and  a built-in 10gb network switch. The FC630 server has 2 - 16 core hyperthreaded processors, giving it the equivalent of 64 processors.  Benchmarks indicate it is about 40% faster than our other cloud servers. Fully populated with 4 FC630 servers, this one piece of equipment would have the equivalent of 256 processors, and could handle all of our current computing needs.

As we upgrade our VMWARE based cloud computing capacity, we are also retiring a number of old servers, and replacing them with servers running in our cloud. The goal is to have almost no physical servers except for our cloud servers and a few machines that serve up attached storage. Both our small HPC grid as well as our hadoop big data cluster will be totally in our cloud, allowing a great deal of flexibility in load balancing. We currently are running at close to 50% cpu (most server environments are closer to 15%) , and  support more than 100 virtual machines running on Windows XP, Windows 7, Windows 10, Windows Server 2008, fedora linux, Centos 5 and 7, Ubuntu 14, ...
These systems connect to about 150TB of SAN and NAS storage.

As we move towards a 10gb backbone, we also have to deconfigure much of our 1gb network. This move has been accelerated by the recent failure of one of our very old extreme switches. As our processing demands grow, the network is becoming the bottleneck. In the next few months, we plan to remove much of our 1gb network equipment, only keeping enough to handle low volume, non-critical needs like access to the administrative ports on equipment. All data will be flowing over our new netgear switches with traffic separated on several VLANs (NAS, ISCSI, VMOTION, VM Management, ...). We hope this will eliminate the occasional storage slowdowns that we are seeing during very high traffic periods.

Future improvements include a higher bandwidth connection to the university research computing facilities, as well as using the  public cloud (probably Amazon EC2) as an expansion capability for short term peak periods.

The goal is to reduce our installed hardware to the bare minimum and have it fully redundant. We are close.




Posted 28th November 2016 by Norman White
Labels: Amazon EC2 big data cloud computing Dell FC630 Dell FX2 HADOOP netgear VLAN vmware
 
12 View comments
OCT
14
It's the cloud, stupid
This week has had several major technology related announcements, all directly related to the rapid switch to cloud computing.

The first is the mega merger of Dell and EMC. This continues and accelerates  the transformation of Dell into an enterprise systems company (where the margins are much higher), and away from the low-end, low margin PC  business. The hidden jewel in the merger is the  shares of VMWARE, the leading cloud software producer, that EMC owns. In fact, the only way the deal was done was by creating tracking stock in VMware which could be sold to generate some of the equity needed. This now creates a company that can produce an integrated cloud computing environment, where Dell can completely control both the hardware and the software. With VMware's VSAN capability, using EMC disk drives and storage devices, Dell will be able to deliver hyperconverged, integrated cloud servers at very attractive prices. (hyperconverged refers to one physical system that includes networking, storage, ram and processing in a single package).

The other recent announcement that impacts cloud computing is intel's latest Haswell processor, which Amazon  AWS already is starting to put in production.  Each processor can run 18 cores, and a server can have 4 cpus, giving it 72 cores, or over the equivalent of over 100 processing threads. Maximum ram is 6TB, which will allow AWS to offer customers 2TB VMs. Intel is Dell's main chip provider, so we can expect to see these chips in Dell servers soon.

A big question is whether Dell has necessary consulting and support capabilities to provide companies private cloud services that can compete against the Amazon and Google public cloud services.  The Cloud Service Providers (CSP) are now having intel design customized chipsets optimized for their environments.  They apparently receive the chipsets months ahead of public release, allowing them to put them into production much earlier than manufacturers like Dell and HP.
The CSPs know the exact environment their processors will run in (as will Dell if it ship integrated cloud servers), so they can deploy them more rapidly.

All of these trends indicate that the move to cloud based computing will continue to be rapid and inevitable, whether it is to private clouds, public clouds, or hybrid clouds.

Posted 14th October 2015 by Norman White
Labels: amazon aws cloud computing could service providers Dell hybrid cloud hyperconverged Intel private cloud public cloud vmware vsan
 
13 View comments
OCT
14
Update on recent facilities upgrades in the Stern Center for Research Computing
There were a number of recent facility and service upgrades that impact research computing users. A list of all of the services and facilities can be found  at http://scrc.stern.nyu.edu

1) Increased/upgraded Storage - We upgraded our storage network with the addition of another 96 Terabytes of storage to handle the rapidly increasing size and number of research data sets.  We now have over 200 TB of disk storage. Our primary storage device was upgraded with solid state disks, to dramatically increase it's performance.

2) Network upgrades - In order to support the rapidly increasing size of research data sets, we have now moved all of our major servers and most of our storage systems onto a 10 gb network backbone, increasing network capacity by a factor of 10. We hope to complete upgrading our remaining storage in the next few months. All of our vmware cloud servers and most of our storage devices are now running at 10 gigabits / second.

3)  NYU Big Data Cluster -The university HPC group  has just deployed a very large hadoop (big data) cluster running the latest version of Cloudera Enterprise, which includes hadoop, hive, pig, spark, impala, oozie, tez, hue, ... The cluster (dumbo) has 44  128gb nodes, 704 cores, and over 1 petabyte of disk storage.  Early benchmarks indicate it is very fast. (Sorts 1 TB of data in about 5 minutes). We are working with the university to provide a high speed link (10gb) between the Stern Research cluster and the NYU/ITS cluster so that researchers can easily move data back and forth between the two facilities. Dumbo also shares very high speed storage with the large HPC cluster (mercer). This will allow researchers to use both clusters to analyze the same data, for instance, running large simulations on the mercer HPC cluster and then using dumbo, the big data cluster to store and analyze/visualize the results.

4)  NYU data center network upgrade -Just announced yesterday, the NYU South Data Center, where our equipment is located, is upgrading it's network backbone to 10 gigabits/sec from 1 gb/sec. We are already in talks about how to use this increased bandwidth to  be able to move data between Stern Research Computing storage devices and the NYU/ITS HPC and big data clusters.
Coming soon...
We are actively working with the  university HPC team to be able to connect SAS, Tableau, and other systems that support a relational data base interface to the new big data cluster.  Hive (an sql like system) will be used to allow remote applications to treat the big data cluster as a large, extremely fast relational data base store. We expect to be able to connect to the new big data cluster both from within the Stern cluster, as well as from the NYU HPC cluster (which now can run SAS jobs). This should bring exciting new capabilities to both Stern and the university.



Posted 14th October 2015 by Norman White
Labels: 10gb big data cloud computing cloudera hive HPC research computing sas tableau vmware
 
4 View comments
MAR
24
Research Computing Move Completed
I am sorry not to have posted  anything for so long, but all of our efforts have been in making the move to the NYU South Data Center. The  physical move went relatively well, considering it was in the middle of a major snow storm. However, we have had a number of issues in adapting to our new network environment, which is considerably more secure that n where we came from. All of our machines now sit behind several levels of firewall, one at the switch level (how many active mac addresses can we have open at any time on a port), one at the local machine firewall (usually IPTABLES),  one at the subnet level  and then one at internet level.

This was challenging because our VMWARE  hosts can have  many mac addresses active on the same switch port, and then have a wide variety of services running on the different machines. In particular, the VMWARE VCENTER server, which is the master host controlling our 5 VMWARE  hosts, has many services running, some locally and some needed for internet access. It took weeks before we finally could get off-site access to it without going through a VPN.

However, in the process of turning off lot's of old equipment and upgrading our VMWARE infrastructure, we now have dramatically increased our processing capabilities with a new Dell R710 server with the equivalent of 40 cores and 256gb of ram, 3 netgear 10gb switches providing a mostly 10gb backbone, and 2 new Infotrend NAS's, one with 2 10gb nics, and the other with 8 1gb nics. We now have well over 100TB of data storage with a high speed network to move things around.

Our "big data" cluster is now running about half of our nodes in our vmware cloud, and the others on older., end of life equipment. We are slowly migrating as much of our processing as possible into our private cloud, with plans of using public clouds for extra processing when necessary. With our high speed cloud storage, we can actually offer data storage in our cloud that is as fast (if not faster) than locally attached disk. We can now support machines with up to 32 processors if necessary and more than 100gb of ram in our cloud. No one has requested anything that large, but my guess is it will happen soon.

More posts coming soon... (I hope)
Posted 24th March 2014 by Norman White
Marcadores: big data  cloud computing armazenamento em  nuvem  nas storage  vmware
 
4 Ver comentários
Carregando
