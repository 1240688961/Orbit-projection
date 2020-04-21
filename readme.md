  这是我用python写的一个脚本,  
可以旋转坐标轴  
似乎没什么用处  
在做轨道投影时，可能会派上用处
需要POSCAR文件，  
算完后会输出直接在屏幕输出新的x,y,z坐标  
使用方法： 在文件夹中同时有CONTCAR 和result.py  
执行
```
./result.py
```
##在下面的代码为主函数，需要根据自己的需求改
```python
if __name__=="__main__":  
    x1,y1,z1=read_CONTCAR()  
    x1,y1,z1=rotate_x_axis(x1,y1,z1,55.476)  
    x1,y1,z1=rotate_z_axis(x1,y1,z1,44.971)	  
    alm=merge_xyz(x1,y1,z1)  
    print(alm)
 ```

这下面段话表示沿着x轴旋转55.476°  ，但具体是顺时针还是逆时针旋转我也忘了，可以自己测试改代码
```
 x1,y1,z1=rotate_x_axis(x1,y1,z1,55.476)
 ```
 这下面段话表示沿着z轴旋转44.971° 
 ```
x1,y1,z1=rotate_z_axis(x1,y1,z1,44.971)	  
 ```
   如需旋转y轴也是同理，这里不再展示
