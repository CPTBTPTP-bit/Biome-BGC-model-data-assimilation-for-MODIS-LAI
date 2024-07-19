#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <iostream>

// [[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
using namespace Eigen;
using namespace std;

// [[Rcpp::export]]
NumericVector DoEnKF(NumericVector a,NumericVector b)
{
  //因为不需要循环了，这里的i/j/m均作废
  //int i,j,m;
  //不循环并不代表不需要j
  int j;
  //double tm[en_num];
  
    typedef VectorXd Dynamic1D;
    //typedef MatrixXd Dynamic2D;
    
    Eigen::Map<Eigen::VectorXd> noise_deviation = as<Eigen::Map<Eigen::VectorXd> >(b);
    
    double  model_noise_deviation,back_noise_deviation,en_num;
    
    model_noise_deviation = noise_deviation(0);
    back_noise_deviation = noise_deviation(1);
    en_num=noise_deviation(3);
    
    Dynamic1D state_ensemble_pre = Dynamic1D::Zero(en_num,1);
    Dynamic1D state_ensemble_post = Dynamic1D::Zero(en_num,1);
    Dynamic1D vector4=Dynamic1D::Zero(en_num,1);
    Dynamic1D vector2=Dynamic1D::Zero(en_num,1);
    Dynamic1D vector1=Dynamic1D::Zero(en_num,1);
    Dynamic1D vector3=Dynamic1D::Zero(en_num,1);
    Dynamic1D returnnumber=Dynamic1D::Zero((en_num+1),1);
    //Dynamic1D state_post = Dynamic1D::Zero(100);
    
    double state_post;
    double ensemble_avg;
    double sum_vector1 = 0;
    //double sum_vector2 = 0;
 
    Dynamic1D sum_mat1 = Dynamic1D::Zero(en_num,1);
    Dynamic1D sum_mat2 = Dynamic1D::Zero(en_num,1);
    //Dynamic1D sum_mat2 = Dynamic1D::Zero(100)
    
    double PHT,HPHT,mat1,mat2,mat3;
    //double guance,sum_mat3,mat4;
    
   
    
    float temp1;
    //,temp2
    
    //string sep = "\n----------------------------------------\n"; 
    //IOFormat CleanFmt(4, 0, " ", "\n"); 
     
    //------------将传递进来的a,传递给VectorXd------并且将前en_num个值赋值给state_ensemble_pre--------
    Eigen::Map<Eigen::VectorXd> ensemble_remote = as<Eigen::Map<Eigen::VectorXd> >(a);
    
    state_ensemble_pre = ensemble_remote.head(en_num);
    

    
    //------------打印矩阵的格式------------------
    
    
    //目的：生成model_noise
    //=======一期实验-16_4_21=============
    
    //模型误差
    NumericMatrix X(en_num,1);
    //观测误差
    NumericMatrix Y(en_num,1);
    
    //下面的Map已经完成了
    //VectorXd model_noise = MatrixXf::Zero(en_num,1);
    
    //打印预测值，然后打印----------------，最后打印state_ensemble_pre
   
    //cout << "预测值"<< sep << state_ensemble_pre.format(CleanFmt) << sep;
    
    

    
    //=======一期实验-16_4_21=============
    //for(j=0;j<100;j++) 
    //{
      //获取模型标准差
      //195-cvmGet(P_measure)
      //工程1，实验阶段，这些值都是模拟的,得到temp2
      //195-相同的j用的一个P_measure的值，得到的随机数赋予了属于同一个j(相同的点)的集合
      //mvnorm,将一个随机正态分布传递给model_noise
      //工程1，可以先模拟model_noise
      
      //=======一期实验-16_4_21=============
      
      // should be replaced by the MODEL
      
      
      
      //产生model噪声，这里假设模型噪声为1
      //这里表示把X的一行变成一个随机的模型的误差
      X(_, 0) = rnorm(en_num,0,model_noise_deviation);
      
      //=======一期实验-16_4_21=============
      //}
      //=======一期实验-16_4_21=============
    
      //Map类通过C++中普通的连续指针或者指针数组来构造Eigen里的Matrix类。这好比Eigen里的Matrix类的数据和raw C++ array共享了一片地址，也就是引用。
      //做模型误差，model_error
      Map<Eigen::VectorXd> model_noise = as<Eigen::Map<Eigen::VectorXd> >(X);
     
      //model_noise应该不需要转换
      // model_noise = model_error.cast<double> ();
      //=======一期实验-16_4_21=============
    
  
    //加模型噪声，求和
    //如果只运行一个点的话，相当于一行求和
    //for(j=0;j<100;j++)
    //{
    //  for(m=0;m<100;m++)
    //  { 
    //假想model_noise是横向的向量,model_noise不是这么简单，是一个以集合数量为行，像元数量为列的数组
    //------vector1(m,0) = model_noise(j,m);-----------
    //}
    //   vector2 = vector1 + state_ensemble_pre.col(j);
      state_ensemble_pre = state_ensemble_pre+model_noise;
      //cout << "state_ensemble_pre"<< sep <<state_ensemble_pre.format(CleanFmt) << sep;
      //cout << "model_noise"<< sep <<model_noise.format(CleanFmt) << sep;
      
      sum_vector1=state_ensemble_pre.sum();
      //cout << "sum_vector1"<< sep <<sum_vector1 << sep;
      
      //sum_vector1 = sum_vector1 + state_ensemble_pre.col(j);
      //}
     
    //求平均值
    //for(j=0;j<100;j++)
    //{
      ensemble_avg = sum_vector1/(double)en_num; 
      
    //}
    
    //EnKF主要过程
    //j还是集合的数目
    //单个点算出来就是一个值！！！
    //for(j=0;j<100;j++)
    //{
      vector4 = state_ensemble_pre.array() - ensemble_avg;
 
      //cout << "PRE-AVG"<< sep <<vector4.format(CleanFmt) << sep;
      //cout << "vector4"<< sep << vector4.format(CleanFmt) << sep;
    
      //196-cvGEMM感觉是乘以H的含义，因为这里没有变化，所以不需要乘以H
      //直接用vector4就行,must be thinked
      //这里有个疑问呢，就是我们考虑的是点和集合数是一样的，如果不一样的情况，还是需要认真考虑下
      
      sum_mat2 = vector4.cwiseProduct(vector4);
      double sum_mat22=sum_mat2.sum();
      
      //cout << "sum_mat2"<< sep <<sum_mat2.format(CleanFmt) << sep;
      //cout << "sum_mat22"<< sep <<sum_mat22 << sep;
      //cout << "mat2"<< sep <<mat3 << sep;
      
      //matPces是什么？可能是后期添加的用来验证的2016-5-18,用来验证，Vector4*Vector4.transpose()与.cwiseProduct的关系
      //matPces = vector4*vector4.transpose();
      //matPces1 = matPces+matPces1;
    
      //sum_mat3和sum_mat2是一回事情，也就是说:不用计算sum_mat3
      //sum_mat3 = vector4.cwiseProduct(vector4) + sum_mat3 ;
      //}
    
    
     // matPces1=matPces1/float(99);
     //cout << "Pf,k+1"<< sep << matPces1.format(CleanFmt) << sep;
    
    
    
    //单个点算出来就是一个值，one number.这里的m是很多点的代表。
    //HPHT和PHT就不是矩阵，就是一个值
    //for(m=0;m<100;m++)
    //{
      //要除以数目减一
      PHT = sum_mat22/double(en_num-1);
      HPHT = sum_mat22/double(en_num-1);
    //}
    
    
    //计算增益矩阵
    //如果是一个点的话，P_measure也是一个单值
    //=======一期实验-16_4_21=============
    //MatrixXf P_measure;
    //=======一期实验-16_4_25,identity to ones========
    //P_measure = MatrixXf::Ones(100,100)*float(19);   //用单位矩阵初始化
    double P_measure = noise_deviation(2);
    //=======一期实验-16_4_21=============
    mat1 = HPHT + P_measure;
    
    //2016-4-25
    //输出HPHT
    //cout << "HPHT"<< sep << HPHT.format(CleanFmt) << sep;
    
    //mat3是gain是一个值，这里mat1是一个double。
    mat2 = 1/mat1;
    //mat3就是卡尔曼增益
    mat3 = PHT * mat2;
    //cout << "HPHT"<< sep <<HPHT << sep;
    //cout << "mat1"<< sep <<mat3 << sep;
    //cout << "mat2"<< sep <<mat3 << sep;
    //cout << "mat3"<< sep <<mat3 << sep;
   // cout << "gain"<< sep <<mat3 << sep;
    //mat4应该等于1
    //mat4 = mat1 * mat2;
    
    //计算diff矩阵中每一列的均值和方差
    //=======一期实验-16_4_21=============
    //VectorXd back_noise = VectorXd::Zero(20,1);
    //=======一期实验-16_4_21=============
    // for(j=0;j<100;j++)
    //{
      /*
      temp2 = P_measure(j,j);
      //设置随机数
    
      for(m=0;m<100;m++)
      { 
        
        //temp1 = Num_col(m,0);
        back_noise(m,j) = temp1;
      }
      */
     // }
  
    
    //=======一期实验-16_4_21=============
   // for(j=0;j<100;j++) 
  //  {
      //获取模型标准差
      //195-cvmGet(P_measure)
      //工程1，实验阶段，这些值都是模拟的,得到temp2
      //195-相同的j用的一个P_measure的值，得到的随机数赋予了属于同一个j(相同的点)的集合
      //mvnorm,将一个随机正态分布传递给model_noise
      //工程1，可以先模拟model_noise
      
      //=======一期实验-16_4_21=============
      Y(_, 0) = rnorm(en_num,0,back_noise_deviation);
      //=======一期实验-16_4_21=============
    //}
    //=======一期实验-16_4_21=============
    Map<Eigen::VectorXd>back_noise = as<Eigen::Map<Eigen::VectorXd> >(Y);
    //back_noise = B_eigen.cast<float> ();
    //复制于88行
    //=======一期实验-16_4_21=============
    
  
    
    //计算更新值，重新考虑下，因为sum_mat1不一定是一个矩阵，而是一列。16-4-26
    sum_vector1=0;
    sum_mat1.setZero();
    
    
    //一期实验：2016-4-23 观测矩阵
    //MatrixXf mo = MatrixXf::Ones(nrow,ncol)*((float)20);  
    //MatrixXi mi = mo.cast<int>();    
    //MatrixXf guance = mi.cast<float>();
    //guance = a[20];
    
    //这样的赋值能否成功?不能成功。
    Dynamic1D guance = Dynamic1D::Zero(en_num,1);
    
    //cout << "观测值"<< sep <<guance.format(CleanFmt) << sep;
    double temp2 = ensemble_remote[en_num];
    guance =  state_ensemble_pre.array()-state_ensemble_pre.array() + temp2;
    
    //打印观测值
    //cout << "观测值"<< sep <<guance.format(CleanFmt) << sep;
    
    //for(j=0;j<100;j++)
    //{
      vector2 = state_ensemble_pre*-1;
      //for(m=0;m<100;m++)
      //{
      //vector3(m,0) = back_noise(j,m);
        //一期实验，en_num16-4-23
      //vectorguance(m,0) = guance(j,m);
      //}
      
      //vector2是读进去的，是固定的，观测值也是固定的，这么大偏差，来源于back_noise
      vector4 = guance + vector2 + back_noise;
      
      //cout << "back_noise"<< sep <<back_noise.format(CleanFmt) << sep;
      //cout << "vector"<< sep <<vector4.format(CleanFmt) << sep;
      
      state_ensemble_post = mat3 * vector4 + state_ensemble_pre;
      // cout << "gain"<< sep <<mat3 << sep;
      //cout << "更新值"<< sep <<state_ensemble_post.format(CleanFmt) << sep;
      //cout << "gain"<< sep <<mat3 << sep;
      
      sum_vector1 = state_ensemble_post.sum();
    //}
    
    // for(j=0;j<100;j++)
    //{
      state_post = sum_vector1/double(en_num);
    //}
    
    //计算误差方差阵
    //for(j=0;j<100;j++)
    //{
    
      vector1 = state_ensemble_post.array() - state_post;
     // for(m=0;m<100;m++)
      //{
        //stdout里面包含了预测值、预测值均值、预测值之差，点m
      //}
      //198-fprintf(stdout,"\n")
      //PrintMat.预测值，POST的均值
      
      //看看后面有没有用到
      vector3 = vector1.cwiseProduct(vector1);
      //mat1这里就是新的误差方差，我认为需要除以en_num
      mat1 = vector3.sum()/en_num;
      //}
      
      //输出更新后的值，将更新后的状态值保存到文件中
      //for(j=0;j<100;j++)
      //{
        //sprintf()。标记日期+集合顺序
        //将state_ensemble_post[]中的结果保存到mod_data中
        //打开today_paramfilename所指向的文件，命名为file3
        //198-输出更新后的状态值，关闭file3
       //}
      
      //在将结果保存到文件前，需要对每个值进行判断
      
       for(j=0;j<en_num;j++)
       {
        //之所以会有m，是以为不同的深度的误差不一样的，m在这里可以理解成3
        //P_measure其实就是3个值。
      //  for(m=0;m<100;m++)
        //{
          //注意，应该是m，j具体还不是很清楚
          temp1 = state_ensemble_post(j);
          if(temp1 < 0)
          {
            state_ensemble_post(j) = state_post;
          }
       }
      
      
      //cout <<"更新值"<< sep << state_ensemble_post.format(CleanFmt) << sep;
      
      //将中间结果保存到文件中
      //当前日期，同化的格点数，同化的集合数目
      //观测数据
      
      /*--------------------输出---------------------------
      for(j=0;j<100;j++)
      {
        //第j+1个集合状态预测值
      }
      
      for(j=0;j<100;j++)
      {
        //第j+1集合个状态更新值
      }
      
      //更新后的状态值,state_post,文件接口file_out
      //将结果保存到DA_result_all文件中，文件接口file(可能)
      //将所有观测值保存到observation中，接口file_obs
      //将所有的vic数据保存到vic中，接口file_vic
      -----------------------输出-------------------*/
      returnnumber(en_num,0) = state_post;
      returnnumber.head(en_num) = state_ensemble_post;
      //cout <<"state_post"<< sep << state_ensemble_post.format(CleanFmt) << sep;
      
      //cout <<"返回值"<< sep << returnnumber.format(CleanFmt) << sep;
      return wrap(returnnumber);
}
       
       
       