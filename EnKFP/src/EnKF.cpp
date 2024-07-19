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
int main()
{
  
  int i,j,m;

  //时间主循环，模拟5天-16-4-20
  for(i=0;i<10;i++)
  {
    //194-sub_measure
    //194-向today_paramfilename读入信息，即file_out
    
    typedef VectorXf Dynamic1D;
    typedef MatrixXf Dynamic2D;
    
    Dynamic2D state_ensemble_pre = Dynamic2D::Zero(100,100);
    Dynamic2D state_ensemble_post = Dynamic2D::Zero(100,100);
    Dynamic1D state_post = Dynamic1D::Zero(100);
    Dynamic1D vector1 = Dynamic1D::Zero(100);
    Dynamic1D vector2 = Dynamic1D::Zero(100);
    Dynamic1D vector3 = Dynamic1D::Zero(100);
    Dynamic1D vector4 = Dynamic1D::Zero(100);
    
    Dynamic1D vectorguance = Dynamic1D::Zero(100);
    
    Dynamic1D  ensemble_avg =  Dynamic1D::Zero(100);
    Dynamic1D  sum_vector1 = Dynamic1D::Zero(100);
    Dynamic1D  sum_vector2 = Dynamic1D::Zero(100);
    Dynamic2D sum_mat1 = Dynamic2D::Zero(100,100);
    Dynamic1D sum_mat2 = Dynamic1D::Zero(100);
    Dynamic1D sum_mat3 = Dynamic1D::Zero(100);
    
    Dynamic2D PHT = Dynamic2D::Zero(100,100);
    Dynamic2D HPHT = Dynamic2D::Zero(100,100);
    Dynamic2D mat1 = Dynamic2D::Zero(100,100);
    Dynamic2D mat2 = Dynamic2D::Zero(100,100);
    Dynamic2D mat3 = Dynamic2D::Zero(100,100);
    Dynamic2D mat4 = Dynamic2D::Zero(100,100);
    
    
    //测试P值，是对角阵或者是什么？
    Dynamic2D matPces = Dynamic2D::Zero(100,100);
    Dynamic2D matPces1 = Dynamic2D::Zero(100,100);
    
    
    float temp1;
    //,temp2
    
    //一期实验-2016-4-23
    int nrow = 100;
    int ncol = 100;
    
    //首先对每个集合进行循环
    //首先预测每个集合的状态量，并求集合的均值
    
    //运行模型，得到分析值的前端
    for(j=0;j<100;j++)
    {
      //根据上一个时刻的状态值预测当前时刻的状态值
      //194-today_paramfilename,ensemble_某个集合_某日
      //195-运行模型：runner
      //将预测结果保存到state_ensemble_pre[j]中
      
      //工程1，实验阶段，这些值都是模拟的,得到mod_data
      
      //194-循环，此时的m应该是grid_num数量
      
      //=======OpenCV版的处理方式===========
      // for(m=0;m<100;m++)
      //{
      //  vector1(m,0) = mod_data(m,0);
      //}
      //state_ensemble_pre.col(j) = vector1;
      //=====================================
      
      //=======一期实验-16_4_21=============
     
      MatrixXf m = (MatrixXf::Random(nrow,ncol)+MatrixXf::Ones(nrow,ncol))*((float)5);  
      //MatrixXi mi = m.cast<int>();    
      
      // Simulated state variable :: should be replaced by the value (LAI) simulated by BGC 
      state_ensemble_pre.col(j) = m.col(j);
      
      //=======一期实验-16_4_21=============
    }
    
    IOFormat CleanFmt(4, 0, " ", "\n");
    //目的：生成model_noise
    
    //=======一期实验-16_4_21=============
    NumericMatrix X(100, 100);
    MatrixXf model_noise = MatrixXf::Zero(100,100);
    string sep = "\n----------------------------------------\n";
    cout << "预测值"<< sep << state_ensemble_pre.format(CleanFmt) << sep;
    //=======一期实验-16_4_21=============
    for(j=0;j<100;j++) 
    {
      //获取模型标准差
      //195-cvmGet(P_measure)
      //工程1，实验阶段，这些值都是模拟的,得到temp2
      //195-相同的j用的一个P_measure的值，得到的随机数赋予了属于同一个j(相同的点)的集合
      //mvnorm,将一个随机正态分布传递给model_noise
      //工程1，可以先模拟model_noise
      
      //=======一期实验-16_4_21=============
      
      // should be replaced by the MODEL
        X(_, j) = rnorm(100,0,2);
      
      //=======一期实验-16_4_21=============
    }
    //=======一期实验-16_4_21=============
    Map<Eigen::MatrixXd> A_eigen = as<Eigen::Map<Eigen::MatrixXd> >(X);
    model_noise = A_eigen.cast<float> ();
    //=======一期实验-16_4_21=============
    
  
    //加模型噪声，求和
    for(j=0;j<100;j++)
    {
      for(m=0;m<100;m++)
      { 
        //假想model_noise是横向的向量,model_noise不是这么简单，是一个以集合数量为行，像元数量为列的数组
        vector1(m,0) = model_noise(j,m);
      }
      vector2 = vector1 + state_ensemble_pre.col(j);
      state_ensemble_pre.col(j) = vector2;
      
      sum_vector1 = sum_vector1 + state_ensemble_pre.col(j);
    }
     
    //求平均值
    for(j=0;j<100;j++)
    {
      ensemble_avg = sum_vector1/(float)100;
    }
    
    //EnKF主要过程
    for(j=0;j<100;j++)
    {
      vector4 = state_ensemble_pre.col(j) - ensemble_avg;
      
      //196-cvGEMM感觉是乘以H的含义，因为这里没有变化，所以不需要乘以H
      //直接用vector4就行,must be thinked
      //这里有个疑问呢，就是我们考虑的是点和集合数是一样的，如果不一样的情况，还是需要认真考虑下
      
      sum_mat2 = vector4.cwiseProduct(vector4) + sum_mat2 ;
      //
      
      matPces = vector4*vector4.transpose();
  
      matPces1 = matPces+matPces1;
    
      //sum_mat3和sum_mat2是一回事情，也就是说:不用计算sum_mat3
      //sum_mat3 = vector4.cwiseProduct(vector4) + sum_mat3 ;
      
    }
    
    
    matPces1=matPces1/float(99);
    cout << "Pf,k+1"<< sep << matPces1.format(CleanFmt) << sep;
    
    
    
    
    
    
    for(m=0;m<100;m++)
    {
      //要除以数目减一
      PHT(m,m) = sum_mat2(m,0)/float(99);
      HPHT(m,m) = sum_mat2(m,0)/float(99);
    }
    
    
    //计算增益矩阵
    //=======一期实验-16_4_21=============
    MatrixXf P_measure;
    //=======一期实验-16_4_25,identity to ones========
    P_measure = MatrixXf::Ones(100,100)*float(19);   //用单位矩阵初始化
    //=======一期实验-16_4_21=============
    mat1 = HPHT + P_measure;
    
    //2016-4-25
    cout << "HPHT"<< sep << HPHT.format(CleanFmt) << sep;
    
    mat2 = mat1.inverse();
    mat3 = PHT * mat2;
    mat4 = mat1 * mat2;
    
    //计算diff矩阵中每一列的均值和方差
    //=======一期实验-16_4_21=============
    MatrixXf back_noise = MatrixXf::Zero(100,100);
    //=======一期实验-16_4_21=============
    for(j=0;j<100;j++)
    {
      /*
      temp2 = P_measure(j,j);
      //设置随机数
    
      for(m=0;m<100;m++)
      { 
        
        //temp1 = Num_col(m,0);
        back_noise(m,j) = temp1;
      }
      */
    }
  
    
    //=======一期实验-16_4_21=============
    for(j=0;j<100;j++) 
    {
      //获取模型标准差
      //195-cvmGet(P_measure)
      //工程1，实验阶段，这些值都是模拟的,得到temp2
      //195-相同的j用的一个P_measure的值，得到的随机数赋予了属于同一个j(相同的点)的集合
      //mvnorm,将一个随机正态分布传递给model_noise
      //工程1，可以先模拟model_noise
      
      //=======一期实验-16_4_21=============
      X(_, j) = rnorm(100,0,2);
      //=======一期实验-16_4_21=============
    }
    //=======一期实验-16_4_21=============
    Map<Eigen::MatrixXd> B_eigen = as<Eigen::Map<Eigen::MatrixXd> >(X);
    back_noise = B_eigen.cast<float> ();
    //复制于88行
    //=======一期实验-16_4_21=============
    
  
    
    //计算更新值，重新考虑下，因为sum_mat1不一定是一个矩阵，而是一列。16-4-26
    sum_vector1.setZero();
    sum_mat1.setZero();
    
    
    //一期实验：2016-4-23 观测矩阵
    MatrixXf mo = MatrixXf::Ones(nrow,ncol)*((float)20);  
    MatrixXi mi = mo.cast<int>();    
    MatrixXf guance = mi.cast<float>();
    
    cout << "观测值"<< sep <<guance.format(CleanFmt) << sep;
    
    
    for(j=0;j<100;j++)
    {
      vector2 = state_ensemble_pre.col(j)*-1;
      for(m=0;m<100;m++)
      {
        vector3(m,0) = back_noise(j,m);
        //一期实验，2016-4-23
        vectorguance(m,0) = guance(j,m);
      }
      
      vector4 = vector3 + vector2 + vectorguance ;
      state_ensemble_post.col(j) = mat3*vector4 + state_ensemble_pre.col(j);
      
      sum_vector1 = state_ensemble_post.col(j) + sum_vector1;
    }
    
    for(j=0;j<100;j++)
    {
      state_post(j,0) = sum_vector1(j,0)/float(100);
    }
    
    //计算误差方差阵
    for(j=0;j<100;j++)
    {
      vector1 = state_ensemble_post.col(j) - state_post;
      for(m=0;m<100;m++)
      {
        //stdout里面包含了预测值、预测值均值、预测值之差，点m
      }
      //198-fprintf(stdout,"\n")
      //PrintMat.预测值，POST的均值
      
      //看看后面有没有用到
      mat1 = vector1.cwiseProduct(vector1);
    }
      
      //输出更新后的值，将更新后的状态值保存到文件中
      for(j=0;j<100;j++)
      {
        //sprintf()。标记日期+集合顺序
        //将state_ensemble_post[]中的结果保存到mod_data中
        //打开today_paramfilename所指向的文件，命名为file3
        //198-输出更新后的状态值，关闭file3
      }
      
      //在将结果保存到文件前，需要对每个值进行判断
      for(j=0;j<100;j++)
      {
        for(m=0;m<100;m++)
        {
          //注意，应该是m，j具体还不是很清楚
          temp1 = state_ensemble_post(m,j);
          if(temp1 < 0)
          {
            state_ensemble_post(m,j) = ensemble_avg(m,0);
          }
        }
      }
      
      cout <<"更新值"<< sep << state_ensemble_post.format(CleanFmt) << sep;
      
      //将中间结果保存到文件中
      //当前日期，同化的格点数，同化的集合数目
      //观测数据
      
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
  }
}