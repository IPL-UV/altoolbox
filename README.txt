MATLAB Active Learning Toolbox for Remote Sensing.

(c) 20011-13 Devis Tuia & Jordi Muñoz-Marí.

This code is licensed under GNU GPL v2.

Instructions
------------

The Active Learning Toolbox (ALTB) works in MATLAB 200x and later versions. The
SVM is solved using an external executable, multisvm, which is based on the
Torch3 library (it is also a very nice OAA SVM for general purpose).

Quick setup:

  - Download last release and install it on your computer. Latest version is at
    * https://altoolbox.googlecode.com/archive/ALToolbox_v0.5.2.zip, or
    * https://github.com/IPL-UV/altoolbox/archive/ALToolbox_v0.5.2.zip
  - Download multisvm_binaries_v2.zip, uncompress it, copy the binary
    corresponding to your platform in the ALTB directory and rename it to
    'multisvm' (or 'multisvm.exe' if you are using Windows).
    URLs:
      * https://github.com/IPL-UV/altoolbox/releases/download/ALToolbox_v0.5.2/multisvm_binaries_v2.zip
      * https://altoolbox.googlecode.com/files/multisvm_binaries_v2.zip
      * https://drive.google.com/open?id=0B4LAJuc0lE8ESUhKN0NaenF5Z1U&authuser=0
  - If you use Windows 64 bits you will also need to download
    mingw64-runtime.zip. Save the contents in the ALTB directory or anywhere in
    your system path.
    URLs:
      * https://github.com/IPL-UV/altoolbox/releases/download/ALToolbox_v0.5.2/mingw64-runtime.zip
      * https://altoolbox.googlecode.com/files/mingw64-runtime.zip
      * https://drive.google.com/open?id=0B4LAJuc0lE8EcUdrdExJalRRckU&authuser=0
  - Run 'demo.m'.

What's next?

The ALTB code is well commented. Read the code in demo.m and in AL.m, which is
the core of the toolbox. If you only want to use its methods, call AL.m with
the appropriate parameters. If you want to include a new method and share it,
look inside AL.m. There is basically a main loop with three steps:

  0: An SVM is trained with a given training/validation set.
  1: The obtained model is used on a set of candidates.
  2: The prediction is ranked according one of the AL methods, a set of samples
     from the candidate set is chosen, and the process is repeated.
     
Be careful, as we divide 'uncertainty' criteria (like margin sampling, i.e. how
uncertain the sample is for the current model) from 'diversity' criteria (like
ABD, i.e. how much selected samples are different between each other). Please
follow this logic in your implementation.

If you develop a new method and want to include it in the library for everyone
to use and test, contact us using the project web page.
     
Compiling multisvm
------------------

If you want to compile multisvm, you will need the Torch3 library. You can
download it from http://www.torch.ch/torch3/downloads.php

Read the instructions to compile the library and install it somewhere in your
system. For multisvm, you need to include the 'core' and 'kernels' modules at
least, but if you want to include all Torch3 modules it won't hurt.

Once you have Torch3 compiled, take a look at the Makefile provided with the
ALTB and modify it for your system. Usually, you only need to change the path
where Torch3 is installed. Then run 'make' in a terminal at it should compile.

References
----------

  - Tuia, D.; Volpi, M.; Copa, L.; Kanevski, M.; Munoz-Mari, J.; , "A Survey of
    Active Learning Algorithms for Supervised Remote Sensing Image
    Classification," Selected Topics in Signal Processing, IEEE Journal of,
    vol.5, no.3, pp.606-617, June 2011. DOI: 10.1109/JSTSP.2011.2139193.
    URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5742970&isnumber=5767682
    
  - Tuia, D.; Muñoz-Marí, J.; , "Learning user's confidence for active
    learning," Geoscience and Remote Sensing, IEEE Transactions on, in press.
    DOI: 10.1109/TGRS.2012.2203605.
    URL:http://ieeexplore.ieee.or/xpls/abs_all.jsp?arnumber=6247502
    
  - Muñoz-Marí, J.; Tuia, D.; Camps-Valls, G.; , "Semisupervised Classification
    of Remote Sensing Images With Active Queries," Geoscience and Remote
    Sensing, IEEE Transactions on , vol.PP, no.99, pp.1-12, 0.
    DOI: 10.1109/TGRS.2012.2185504.
    URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6166388&isnumber=4358825 

Thanks
------

  - Fred Ratle, for developing and sharing multisvm.
  - Michele Volpi, coder of GridSearchTrain_CV.m.
  - Lexie Yang, tester and developer.

