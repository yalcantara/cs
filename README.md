# cs

A complete primitives of Matrix, Vectors & operations to work with CPU and GPU workloads.


```C++

//creates the memomy buffer on GPU and transfer the data.
GpuMatrix a = { { 1, 2, 3 }, { 4, 5, 6 } };  
GpuMatrix b = { { 1, 2 }, { 3, 4 }, { 5, 6 } };
	
auto scal = 2 * a; //operator overloading ^_^
scal.print();
	
auto d = a.dot(b);
d.print();
GpuVector v = { 1, 2 };
auto c = a.affine(b, v);
	
//transfer the data back to main memory and send it to the console
c.print(); 
```
