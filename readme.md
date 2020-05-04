这是我用python写的一个脚本,  
可以用来做轨道投影，  
PROCAR里面是包含了投影的结果  
速度很慢,我测试过代码  
当绘制3600个点时, 90%的时间花费在scatter上  
因此, 优化scatter是很难的  
只能用来做考虑自旋的体系，即INCAR中，ISPIN=2    
当然ISPIN=1时，稍微改下也能用   
在做轨道投影时，可能会派上用处   
需要CONTCAR, KPOINTS, PROCAR, EIGENVAL文件，   
算完后会输出直接在屏幕输出图片，并自动保存band.eps矢量图     
使用方法： 在文件夹中同时包含上述文件    
执行  
```
./result_last.py
```

下面讲解下代码    
下面这些一定要填，reference_level，表示以哪个能级作为0势能参考点，   
kpath表示你在算band时的KPOINTS的路径   
y_axis_min，y_axis_max表示绘制出来的y轴范围   

```
reference_level = -2.6880
kpath = 'GXSYG'
y_axis_min = -5
y_axis_max = 3
```

##在下面的代码为主函数，需要根据自己的需求改

```
if __name__ == '__main__':


    # Plot the need orbit and atoms

 ```
    下段这里，i表示要投影的原子序号，
    dict1['dz2']表示要投影的哪个轨道
    1000代表投影出来的点的放大倍数
    red,skyblue表示用什么颜色
 ```
    i = 1
    procar_plt_up(i, dict1['dz2'], 1000, 'red')
    procar_plt_down(i, dict1['dz2'], 1000, 'skyblue')
 ```
    # Set the range of abscissa and ordinate
    plt.xlim(min(axis_x), max(axis_x))
    plt.ylim(y_axis_min, y_axis_max)
    plt.yticks(np.arange(y_axis_min, y_axis_max, 0.2), size=25)
    # Set the xlabel and xylabel  and title
    # plt.ylabel("E-E_VBM(eV)", size=25)
```
    这下面段话表示，用于右上角的图注,s表示图注的大小，alpha表示透明度，label就是图注中的文字
```
    # Set the legend
    ax.scatter(10000, 10000, s=240, c='red', edgecolor='none', alpha=1.0, label='Ni-dxz_up')
    ax.scatter(10000, 10000, s=240, c='skyblue', edgecolor='none', alpha=1.0, label='Ni-dxz_down')

 ```
    font1 = {'weight': 'normal', 'size': 20}
    legend = plt.legend(prop=font1)
    plt.savefig('band.eps')
    # os.system("convert -density 288 band.eps band.png")
    plt.show()
```

 这下面段话表示将矢量图转换成png格式的图片，必要去掉注释就行
 ```
os.system("convert -density 288 band.eps band.png")
 ```
当然如果投影的原子很多，在i的变量上写个循环就可以了  
不过速度很慢  
当然也可以根据需要进行优化  
