# Assignment 03

Task 1:

How to run my SLIC implementation:
	Open main.py

	In the parse_args() method adjust filename to whatever image you want to use. Make sure to include the file path if necessary.
	The default image is of the Shapiro Science Center at Brandeis
	You can also adjust epochs (iterations), clusters (number of clusters used), and m (compactness) to see the difference

	Click run

	Each iteration of the SLIC method will output the time it took and what iteration it is on.
	Each iteration will also output which cluster number was not used in the past iteration. (If there is any)

	After completing the partitions it will output 4 images. They will appear in this order
	1. Image of the partitions alone
	2. Image of the original image segmented by my partitions using my method
	3. Image of the original image segmented by my partitions using the Scikit method
	4. The original image

How to run the Scikit SLIC implementation:
	Open scikittest.py

	You can adjust the,
		file name (default is "brandeis.jfif")
		n_segments (number of segments used)
		enforce_connectivity (default is true)	

Task2:
	Demo 1:

	Open command prompt and enter python main.py demo1 -i samples/brandeis.jfif -a resnet152 -t layer4

	Make sure the path is correct and it should run and the resulting images should end up in results folder 

	To use other images adjust samples/brandeis.jfif

	Demo 2:

	Adjust the number in line 280 in main.py in grad-cam-pytorch-master to what classification id you want to use

	Open command prompt and enter python main.py demo2 -i samples/brandeis.jfif

	Make sure the path is correct and it should run and the resulting images should end up in results folder 

	To use other images adjust samples/brandeis.jfif