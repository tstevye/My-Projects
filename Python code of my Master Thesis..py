#Creator: Stevye F. Tinchie
#Purpose: Master Thesis
#Period: October 2016 - July 2017

import scipy.sparse.linalg
from math import *
from sympy import *
import numpy as np
from scipy import interpolate
from numpy import *
import sys
import matplotlib.tri as mtri
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
#ax = Axes3D(figure())
#fig = plt.figure(1)
#ax = fig.gca(projection='3d')

print('PYTHON SOFTWARE TO NUMERICALLY SOLVE THE POISSON EQUATION ON A POLYGONAL DOMAIN USING THE FIVE AND THE NINE POINT FINITE DIFFERENCE')
print('')
D=int(input('\033[1;34mPRESS\033[1;34m \n\033[1;38m  5:FOR THE FIVE POINT FINITE DIFFRERENCE \n\n 9: FOR THE NINE POINT FINITE DIFFERENCE \n \033[1;38m'));q=D
if(D==5):
    print('WELCOME TO THE METHOD OF FINITE DIFFERENCE USING THE 5-POINT STENCIL')
    # ********************************************************FUNCTION 1******************************************************************
    # ************************************************************************************************************************************
    # The function InputDomain request the user to input the cordinates of his/her desired polygonal domain anti-clockwise and plots the corresponding geometry
    print('')
    def InputDomain():
        A = n = int(input('WHAT IS THE NUMBER OF SIDES YOU WISH TO CONSIDER TO MAKE UP YOUR POLYGONAL DOMAIN ?' + ' '))
        print('PLEASE CHOOSE HOW YOU WANT TO COLLECT THE INPUT INFORMATION')
        qa = int(input('\033[1;34mPRESS \033[1;34m\n\033[1;38m1:TO COLLECT THE INPUT MANUALLY\033[1;38m\n\n2:TO COLLECT THE INPUT FROM A FILE\n'))
        if qa == 1:
            x = []; y = []; i = 1;
            while (i < n + 1):
                x1 = float(input('x' + str([i]) + '='));y1 = float(input('y' + str([i]) + '='))
                x.append(x1);y.append(y1);i += 1
            print('x = ',x);print('y = ',y) ; f1 = str(input('ENTER THE SOURCE FUNCTION, f(x,y) = '))
            q = input('ENTER THE BOUNDARY FUNCTION, B(x,y)=  ') ; U1 = input('ENTER THE TRUE SOLUTION, u(x,y)=  ')
        else:
            Results = input('Enter the path to the file \n')
            Sheet = open(Results + '.txt', 'r')
            x1 = []
            readFile = open(Results + '.txt', 'r')
            sepFile = readFile.read().split('\n')
            # print(sepFile)
            readFile.close()
            for plotFair in sepFile:
                xa = plotFair.split(',')
                for xb in xa:
                    xc = xb.split(',')
                    x1.append((xc[0]))
            x1.remove('');x=[float(x1[0]),float(x1[2]),float(x1[4]),float(x1[6])]
            y = [float(x1[1]), float(x1[3]), float(x1[5]), float(x1[7])];f1 = x1[8];q = float(x1[9]) ; U1 = x1[10]
            print('x = ',x) ; print('y = ',y)
        #plt.xlabel('x-axis', color='red');plt.ylabel('y-axis', color='red');
        x.append(x[0]);y.append(y[0]);
        plt.xlabel('x-axis', color='red');plt.ylabel('y-axis', color='red');plt.plot(x, y);
        plt.savefig('Polygonal_Domain.png', transparent=True)
        plt.title('This is a plot of the polygonal domain', color='red')
        plt.show()
        return (x,y,n,f1,q,U1)
    x, y, n, f1,q,U1 = InputDomain()
    print('')

    # ********************************************************FUNCTION 2******************************************************************
    # ************************************************************************************************************************************
    # Function 3 calculates the midpoints of the given geometry
    def Midpoints(x, y, n):
        Mx = []; My = []
        for j in range(0, n - 1):
            for k in range(j + 1, n):
                x2 = (x[k] + x[j]) / 2;
                y2 = (y[k] + y[j]) / 2
                Mx.append(x2);
                My.append(y2)
        return (Mx, My)
    Mx, My = Midpoints(x, y, n)
    print('')

    # ----------------------------------------------------FUNCTION 3---------------------------------------
    # ------------------------------------------------------------------------------------------------

    # Function 3 identifies the ccordinates of the calculated midpoints
    def ReorganizingNodes(Mx, My):
        Mxx = [Mx[0], Mx[1], Mx[2], Mx[3], Mx[4], Mx[5]]
        Myy = [My[0], My[1], My[2], My[3], My[4], My[5]]
        #print('Mxx=',Mxx); print('Myy=', Myy)
        return (Mxx, Myy)
    Mxx, Myy = ReorganizingNodes(Mx, My)
    print('')
    # -----------------------------------------------------FUNCTION 4--------------------------------------
    # ------------------------------------------------------------------------------------------------

    # Function 4 performs and plots the first discretization of the given doamin
    def SmallerRecs(x, y, Mxx, Myy):
        CNodes = [];
        CNodesNumb = []
        s1 = [x[0], Mxx[0], Mxx[1], Mxx[2], x[0]], [y[0], Myy[0], Myy[1], Myy[2], y[0]];
        s2 = [Mxx[0], x[1], Mxx[3], Mxx[4], Mxx[0]], [Myy[0], y[1], Myy[3], Myy[4], Myy[0]]
        s3 = [Mxx[2], Mxx[1], Mxx[5], x[3], Mxx[2]], [Myy[2], Myy[1], Myy[5], y[3], Myy[2]];
        s4 = [Mxx[4], Mxx[3], x[2], Mxx[5], Mxx[4]], [Myy[4], Myy[3], y[2], Myy[5], My[4]]
        s = [s1, s2, s3, s4]
        #print('s=', s)
        plt.plot(s1[0], s1[1], s2[0], s2[1], s3[0], s3[1], s4[0], s4[1], color='magenta');
        plt.title('This is the First discretization', color='red');plt.xlabel('x-axis', color='red');plt.ylabel('y-axis', color='red')
        plt.savefig('First dis.png', transparent=True)
        plt.show()
        return (s, s1, s2, s3, s4, CNodes, CNodesNumb)
    s, s1, s2, s3, s4, CNodes, CNodesNumb = SmallerRecs(x, y, Mxx, Myy)
    print('')

    # ----------------------------------------------FUNCTION 5-------------------------------------------
    # -----------------------------------------------------------------------------------------------

    # function 5 requests the user to specify the number of partitions he/she will like to consider
    def NumPartition():
        BB = int(input('\033[1;34mPRESS  2\033[1;34m \n  TO DEFINE THE NUMBER OF PARTITIONS OF YOUR DOMAIN:\n'))
        if (BB == 2):
            E = int(input('Enter your desired number of partition along the x-axis:\n n =  '))
            while (E == 0 or E < 0):
                print('\033[1;31m!!!n should be a natural number different from 0!!!\033[1;31m')
                E = int(input('\033[1;38mn=\033[1;38m'))
            F = int(input('Enter yor desired number of partition along the y-axis:\n m =  '))
            while (F == 0 or F < 0):
                print('\033[1;31m!!!m should be a natural number different from 0!!!\033[1;31m')
                F = int(input('\033[1;38m m=\033[1;38m'))
            print("")
        else:
            while (BB != 2):
                print('******ERROR******: PLEASE MAKE SURE YOU ENTER 2.')
                BB = int(input('\033[1;34mPRESS  2\033[1;34m \n  TO DEFINE THE NUMBER OF PARTITIONS OF YOUR DOMAIN:\n'))
                E = int(input('Enter your desired number of partition along the x-axis:\n n =  '))
                while (E == 0 or E < 0):
                    print('\033[1;31m!!!n should be a natural number different from 0!!!\033[1;31m')
                    E = int(input('\033[1;38mn=\033[1;38m'))
                F = int(input('Enter yor desired number of partition along the y-axis:\n m =  '))
                while (F == 0 or F < 0):
                    print('\033[1;31m!!!m should be a natural number different from 0!!!\033[1;31m')
                    F = int(input('\033[1;38mm=\033[1;38m'))
                print("")
        return (E, F, BB)
    E, F, BB = NumPartition()
    print('')


    # ----------------------------------------------FUNCTION 6-------------------------------------------
    # -----------------------------------------------------------------------------------------------

    # Function 6 partions the interval of the domain in subintervals of size h and k
    def IntervalPart(E, F, x, y):
        h = round((x[1] - x[0])/E, 8)
        k = round((y[3] - y[0])/F, 8)
        print('h=', h);
        print('k=', k)
        return (h, k)
    h, k = IntervalPart(E, F, x, y)
    print('')
    # ----------------------------------------------FUNCTION 7-------------------------------------------
    # -----------------------------------------------------------------------------------------------

    # Function 7 defines the entire mesh grid including the boundary points
    def TotalgridPoints(x, y, E, F):
        r = [];
        s = []
        for j in range(0, F + 1):
            for i in range(0, E + 1):
                GB = (x[0] + i * h, y[0] + j * k)  # definining mesh points
                r.append(GB[0])
                s.append(GB[1])
        print('r =', r);
        print('s =', s)
        return (r, s)
    r, s = TotalgridPoints(x, y, E, F)
    print('')

    # ----------------------------------------------FUNCTION 7-------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # Function 8 initializes a matrix whose entries are zero
    def MatrixInitialize2(E, F):
        L = np.zeros([(E + 1) * (F + 1), (E + 1) * (F + 1)])
        # B = reshape(A,[(E+1)*(F+1)-1,(E+1)*(F+1)-1])
        return (L)
    L = MatrixInitialize2(E, F)
    print('')
    #----------------------------------------------FUNCTION 8-------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #Function 8 modifies the diagonal and the under diagonal elements
    def ModifyDiagon(E,F,h,k,L):
      i = 0
      while(i < (E+1)*(F+1)):
        p1 = 2*(h*h + k*k)
        p2 = (h*h)*(k*k)
        L[i][i]=p1/p2
        i+=1
      return(L)
    L = ModifyDiagon(E,F,h,k,L)
    print('')

    #----------------------------------------------FUNCTION 9-------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #Function 9 modifies the under diagonal elements different from zero
    def ModifyUnderDia(E,F,h,k,r,s):
          for j in range (0,(E+1)*(F+1)):
            for i in range(j+1,(E+1)*(F+1)):
              if (round((r[i]-r[j]),4) <= h and s[i]-s[j] == 0):
                L[i][j] = L[j][i] = (-1)*(pow(h,-2))#(-1)*(h**(-2))
              elif (r[i]-r[j] == 0 and round((s[i]-s[j]),4) <= k):
                L[i][j] = L[j][i] = (-1)*(pow(k,-2))#(-1)*(k**(-2))
          return(L)
    L = ModifyUnderDia(E,F,h,k,r,s)
    print('')

    #----------------------------------------------FUNCTION 10-------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #Function 10 resuest the user for a source term and the boundary condition
    def SourceFunx(E,F,r,s,f1,q,U1):
      CC = int(input('\033[1;34mPRESS  3\033[1;34m \n  TO INPUT YOUR TRUE SOLUTION, THE SOURCE FUNCTION AND THE BOUNDARY FUNCTION:\n'))
      if (CC == 3):
        #print(U1)
        U = U1
        Ts = []; intn = []
        for j in range(1, E):
          for i in range(1, F):
            x = r[i + j * (E + 1)]
            y = s[i + j * (E + 1)]
            intn.append([x,y])
            Ts.append(((eval(str(U1)))))
        print("Ts = ", Ts)
        print('intnodes = ', intn)
        print('')

        f = eval(f1)
        v = []
        for j in range(1, E):
          for i in range(1, F):
            x = r[i + j * (E + 1)]
            y = s[i + j * (E + 1)]
            (eval(str(f1)))
            v.append(float((eval(str(f1)))))
        print('v = ', v)
        print('')
        u=eval(str(q))
      else:
        # print('ERROR')
        while (CC != 3):
          print('******ERROR******, PLEASE MAKE SURE YOU ENTER 3.')
          CC = int(input('\033[1;34mPRESS  3\033[1;34m \n  TO INPUT YOUR TRUE SOLUTION, THE SOURCE FUNCTION AND THE BOUNDARY FUNCTION:\n'))
          # if(CC==3):
        Ts = []
        for j in range(1, E):
          for i in range(1, F):
            x = r[i + j * (E + 1)]
            y = s[i + j * (E + 1)]
            (eval(str(U1)))
            Ts.append(float((eval(str(U1)))))
        print("Ts = ", Ts)

        f = eval(f1)
        v = []
        for j in range(1, E):
          for i in range(1, F):
            x = r[i + j * (E + 1)]
            y = s[i + j * (E + 1)]
            #g=(eval(str(f))) ;print('g=', g); g1 = round(g, 4) ; print('g1=', g1)
            v.append(float(eval(str(f1))))
        print('v = ', v)
        print('')
        #u = eval(q)
        print('Boundary segment 1 Numbering')
        b1 = []
        for i in range (0,E):
          x = r[i]
          y = s[i]
          #(eval(u))
          b1.append(float((eval(q))))
          #print(b1)

        #print('Boundary segment 2 Numbering')
        b2 = []
        for j in range (0,F+1):
          i =  j + (j+1)*E
          x = r[i]
          y = s[i]
          #(eval(u))
          b2.append(float((eval(q))))
          #print(b2)

        #print('Boundary segment 3 Numbering')
        b3 = []
        for i in range (0,E):
          j =  i + (E+1)*F
          x = r[j]
          y = s[j]
          #(eval(u))
          b3.append(float((eval(q))))
          #print(b3)

        #print('Boundary segment 4 Numbering')
        b4 = []
        for j in range (1,F):
          i =  j + j*E
          x = r[i]
          y = s[i]
          #(eval(u))
          b4.append(float((eval(q))))
          #print(b4)
      return(Ts,f,u,v,x,y,CC,U)
    Ts,f,u,v,x,y,CC,U = SourceFunx(E,F,r,s,f1,q,U1)
    print('')

    #----------------------------------------------FUNCTION 11-------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #Function 11 resuest the user for a source term and the boundary condition
    def SystemMatrix():
      (A) = ModifyDiagon(E,F,h,k,L)
      #print('System Matrix = ',A)
      return((A))
    (A) = SystemMatrix()
    print('')

    #----------------------------------------------FUNCTION 12-------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #Function 12 computes the reduced matrix
    def ReducedMatrix(E,F):
              print("REDUCED MATRIX")
              A = L
              s1 = (E + 1) * (F + 1)
              for j in range(0, F + 2):
                j = 0
                A = np.delete(A, (j), axis=0)
                A = np.delete(A, (j), axis=1)

              k1 = s1 - 1 - (F + 2)
              for j in range(0, F + 2):
                A = np.delete(A, (k1), axis=0)
                A = np.delete(A, (k1), axis=1)
                k1 = k1 - 1

              for k1 in range(1, E - 1):
                for j in range(0, 2):
                  A = np.delete(A, k1 * (E - 1), axis=0)
                  A = np.delete(A, k1 * (E - 1), axis=1)
              print(A)
              return(A)
    A = ReducedMatrix(E,F)
    print('')

    #----------------------------------------------FUNCTION 13-------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #Function 13 computes the correction of the right hand side using the boundary
    def Correction(E, F, r, s, v, u):
        # print('CORRECTING THE RIGHT HAND SIDE USING BOUNDARY SEGMENTS')
        # print("WITH BOUNDARY SEGMENT 1")
        print()
        b = []
        p1 = 0
        for j in range(1, F):
            for i in range(1, E):
                x = r[i]
                y = s[i]
                q1 = i + j * (E + 1)
                q2 = float(eval(str(u)))
                v[p1] = v[p1] - (q2 * L[q1][i])
                b.append(v[p1])
                p1 += 1
                # print(b)
        print('')
        # print("WITH BOUNDARY SEGMENT 2")
        b = []
        p2 = 0
        for j in range(1, F):
            for i in range(1, E):
                x = r[E + j * (E + 1)]
                y = s[E + j * (E + 1)]
                q3 = i + j * (E + 1)
                q4 = E + j * (E + 1)
                q5 = float(eval(str(u)))
                v[p2] = v[p2] - (q5 * L[q3][q4])
                b.append(v[p2])
                p2 += 1
                # print(b)
        print('')
        # print("WITH BOUNDARY SEGMENT 3")
        b = []
        p3 = 0
        for j in range(1, F):
            for i in range(1, E):
                x = r[j * (E + 1)]
                y = s[j * (E + 1)]
                q6 = i + j * (E + 1)
                q7 = j * (E + 1)
                q8 = float(eval(str(u)))
                v[p3] = v[p3] - (q8 * L[q6][q7])
                b.append(v[p3])
                p3 += 1
                # print(b)

        print('')
        # print("WITH BOUNDARY SEGMENT 4")
        b = []
        p4 = 0
        for j in range(1, F):
            for i in range(1, E):
                x = r[j * (E + 1)]
                y = s[j * (E + 1)]
                q9 = i + j * (E + 1)
                q10 = j * (E + 1)
                q11 = float(eval(str(u)))
                v[p4] = v[p4] - (q11 * L[q9][q10])
                b.append(v[p4])
                p4 += 1
                # print(b)
        print('b = ', b)
        return (x, y, b)
    x, y, b = Correction(E, F, r, s, v, u)
    print('')

    import numpy as np
    def true(x,y):
        TS1 = (x ** 2 - 1) * (y ** 2 - 1) * exp(x * y)
        return TS1
    TS1 = true(x,y)

    def App_Solution(A,E,F,v,b,Ts,r,s,TS1):
        # exact solution plot
        f = plt.figure()
        a = f.gca(projection='3d')
        x = np.arange(-1, 1, 0.01)
        y = np.arange(-1, 1, 0.01)
        X, Z = np.meshgrid(x, y)
        u = (x**2-1)*(y**2-1)*exp(x*y) #u = (x**2-1)**2*(y**2-1)**2*sin(x)*cos(y)
        s = a.plot_surface(X, Z,u, cmap=cm.jet)
        f.colorbar(s, shrink=0.5);
        plt.savefig('True_solution2.png', transparent=True)
        plt.title('This is a plot of the true solution2', color='red')
        plt.xlabel('x-axis', color='red');
        plt.ylabel('y-axis', color='red')
        plt.show()

        print('\033[1;35m!!!! WELCOME TO OUR PROGRAM FOR SOLVING SYSTEMS OF EQUATIONS (Ax=b) USING EITHER A DIRECT METHOD OR AN ITERATIVE METHOD !!!!\033[1;35m\n\n')
        DD = int(input('\033[1;34mPRESS\033[1;34m \n 4: TO ACCESS AND CHOOSE YOUR DESIRED SOLVER:\n'))
        D = int(input('\033[1;34mPRESS\033[1;34m \n\033[1;38m  1:FOR A DIRECT METHOD \n\n 2: FOR AN ITERATIVE METHOD \n \033[1;38m'));
        q = D
        if (D == 1):
            D = int(input('\033[1;34mPRESS\033[1;34m \n \033[1;38m 1:FOR QR DECOMPOSITION\033[1;38m \033[1;31m)\033[1;31m\n\n \033[1;38m 2:FOR LU FACTORIZATION\033[1;38m\033[1;31m\n\n \033[1;38m 3:FOR CHOLESKY FACTORIZATION\033[1;38m \n'))
            q1 = D
            if (D == 1):
                print('!!\033[1;33mYOUR CHOICE IS QR DECOMPOSITION!!\033[1;33m')
            elif (D == 2):
                print('!!\033[1;33mYOUR CHOICE IS LU DECOMPOSITION!!\033[1;33m')
            elif (D == 3):
                print('!!\033[1;33mYOUR CHOICE IS CHOLESKY DECOMPOSITION!!(Small tip:This decomposition can be used as a test for positive definity and symmetry of a matrix)!!\033[1;33m')
            else:
                print('\033[1;31m!!!!!!!!!ERROR MESSAGE!!!!ENTER EITHER 1,2 OR 3 TO MAKE A CHOICE FOR YOUR DIRECT METHOD!!!!!!!!!\033[1;31m')
                exit()
        elif (D == 2):
            D = int(input('\033[1;34mPRESS\033[1;34m \n \033[1;38m 1:FOR JACOBI ITERATION METHOD \n\n 2:FOR GAUSS-SEIDEL ITERATION METHOD \n\n 3:FOR CONJUGATE GRADIENT METHOD \033[1;38m \n'))
            q1 = D

            if (D == 1):
                print('!!\033[1;33mYOUR CHOICE IS JACOBI ITERATION METHOD\033[1;33m!!')
            elif (D == 2):
                print('!!\033[1;33mYOUR CHOICE IS GAUSS-SEIDEL ITERATION METHOD\033[1;33m!!')
            elif (D == 3):
                print('!!\033[1;33mYOUR CHOICE IS CONJUGATE GRADIENT METHOD\033[1;33m!!')
            else:
                print('\033[1;31m!!!!!!!!ERROR MESSAGE!!!!YOU SHOULD ENTER EITHER 1 OR 2 TO MAKE A CHOICE\033[1;31m!!!!!!!!')
                exit()
        else:
            print('\033[1;31m!!!!!!!ERROR MESSAGE!!!!!ENTER EITHER 1 OR 2 TO CHOOSE A METHOD.!!!!!\033[1;31m')
            exit()
        print('\n')

        # SOLVING SYSTEM QR DECOMPOSITION
        print("\033[1;34mQR DECOMPOSITION\033[1;m")
        Q, R = scipy.linalg.qr(A)
        '''
        print("A:")
        pprint.pprint(A)

        print("Q:\n")
        pprint.pprint(Q)

        print("R:\n")
        pprint.pprint(R)

        A = dot(Q,R) #verifying that A = QR
        print("A:")
        pprint.pprint(A)

        print("b:")
        pprint.pprint(b)
        '''
        print()

        def QRDecomposition(A, b):
            Q, R = scipy.linalg.qr(A)
            y = scipy.linalg.solve(Q, b)
            #print(y)
            xs = scipy.linalg.solve(R, y)
            print('\033[1;48mQRDecomposition result is xs = \n %s\033[1;m' % xs)
            i = 0;j = 0;zzss = []
            while (i < len(xs) and j < len(Ts)):
                zzs = (Ts[j] - xs[i]);
                zzss.append(zzs)
                i += 1;j += 1
            #print('zzss = ', zzss)
            error1 = np.linalg.norm((zzss), ord=inf, axis=None, keepdims=False)
            print("\033[1;32m Error1= \033[1;m:", error1)
            error2 = maximum.reduce(fabs(zzss))
            print("\033[1;32m Error2= \033[1;m:", error2)
            rate = log2(0.000244431588331 / 6.11172499153e-05);
            print('rate = ', rate)
            return xs,error1,error2

        # x = QRDecomposition(A,b)
        print("***************************************************************************************")
        print("***************************************************************************************")

        # SOLVING SYSTEM USING LU DECOMPOSITION
        print("\033[1;34mLU DECOMPOSITION\033[1;m")

        def LUFactorization(A, b):
            P, L, U = scipy.linalg.lu(A)
            y = np.linalg.solve(L, b);
            xs = np.linalg.solve(U, y)  # Forward substitution and backward elimination
            print('\033[1;48mLUFactorization result is xs = \n %s\033[1;m' % xs)
            i = 0;
            j = 0;
            zzss = []
            while (i < len(xs) and j < len(Ts)):
                zzs = (Ts[j] - xs[i]);
                zzss.append(zzs)
                i += 1;
                j += 1
            #print('zzss = ', zzss)
            error1 = np.linalg.norm((zzss), ord=inf, axis=None, keepdims=False)
            print("\033[1;32m Error1= \033[1;m:", error1)
            error2 = maximum.reduce(fabs(zzss))
            print("\033[1;32m Error2= \033[1;m:", error2)
            rate = log2(0.000244431588331 / 6.11172499153e-05);
            print('rate = ', rate)
            return xs,error1,error2

        # x = LUFactorization(A, b)
        print("***************************************************************************************")
        print("***************************************************************************************")

        # SOLVING THE SYSTEM USING CHOLESKY DECOMPOSITION
        print("\033[1;34mCHOLESKY DECOMPOSITION\033[1;m")

        def CholeskyDecomposition(A, b):
            L = scipy.linalg.cholesky(A, lower=True);
            LT = scipy.linalg.cholesky(A, lower=False)
            # print(L);print(LT);
            y = np.linalg.solve(L, b);
            xs = np.linalg.solve(LT, y)
            print('\033[1;48mCholeskyDecomposition result is xs = \n %s\033[1;m' % xs)
            i = 0;
            j = 0;
            zzss = []
            while (i < len(xs) and j < len(Ts)):
                zzs = (Ts[j] - xs[i]);
                zzss.append(zzs)
                i += 1;
                j += 1
            #print('zzss = ', zzss)
            error1 = np.linalg.norm((zzss), ord=inf, axis=None, keepdims=False)
            print("\033[1;32m Error1= \033[1;m:", error1)
            error2 = maximum.reduce(fabs(zzss))
            print("\033[1;32m Error2= \033[1;m:", error2)
            rate = log2(0.000244431588331 / 6.11172499153e-05);
            print('rate = ', rate)
            return xs,error1,error2

        # x = CholeskyDecomposition(A,b)
        print("***************************************************************************************")
        print("***************************************************************************************")

        ITERATION_LIMIT = 1000

        # ITERATIVE METHODS.
        # SOLVING SYSTEM USING JACOBI ITERATION
        def JacobiIteration(A, b):
            print('\033[1;34m  JACOBI ITERATION\033[1;m')
            '''
            print("System:")
            for i in range(A.shape[0]):
                row = ["{}*xs{}".format(A[i, j], j + 1) for j in range(A.shape[1])]
                print(" + ".join(row), "=", b[i])
                '''
            print()

            xs = np.zeros_like(b)
            for it_count in range(ITERATION_LIMIT):
                print("Current solution:", xs)
                xs_new = np.zeros_like(xs)

                for i in range(A.shape[0]):
                    s1 = np.dot(A[i, :i], xs[:i])
                    s2 = np.dot(A[i, i + 1:], xs[i + 1:])
                    xs_new[i] = (b[i] - s1 - s2) / A[i, i]

                if np.allclose(xs, xs_new, atol=1e-10):
                    break

                xs = xs_new

            print("Solution, xs =", xs)
            print("")
            i = 0;
            j = 0;
            zzss = []
            while (i < len(xs) and j < len(Ts)):
                zzs = (Ts[j] - xs[i]);
                zzss.append(zzs)
                i += 1;
                j += 1
            # print('zzss = ', zzss)
            error1 = np.linalg.norm((zzss), ord=inf, axis=None, keepdims=False)
            print("\033[1;32m Error1= \033[1;m:", error1)
            error2 = maximum.reduce(fabs(zzss))
            print("\033[1;32m Error2= \033[1;m:", error2)
            rate = log2(0.000244431588331 / 6.11172499153e-05);
            print('rate = ', rate)
            return xs, error1, error2


        # x,error = JacobiIteration(A,b)

        print("***************************************************************************************")
        print("***************************************************************************************")
        print('\n\n')

        # SOLVING SYSTEM USING GAUSS-SIEDEL ITERATION
        def GaussSiedelIteration(A, b):
            print('\033[1;34m  GAUSS-SEIDEL ITERATION\033[1;m')

            ITERATION_LIMIT = 1000
            # prints the system
            '''
            print("System:")
            for i in range(A.shape[0]):
                row = ["{}*xs{}".format(A[i, j], j + 1) for j in range(A.shape[1])]
                print(" + ".join(row), "=", b[i])
                '''
            print()

            xs = np.zeros_like(b)
            for it_count in range(ITERATION_LIMIT):
                print("Current solution:", xs)
                xs_new = np.zeros_like(xs)

                for i in range(A.shape[0]):
                    s1 = np.dot(A[i, :i], xs_new[:i])
                    s2 = np.dot(A[i, i + 1:], xs[i + 1:])
                    xs_new[i] = (b[i] - s1 - s2) / A[i, i]

                if np.allclose(xs, xs_new, rtol=1e-8):
                    break

                xs = xs_new

            print("Solution,xs = ", xs)
            print("")
            i = 0;
            j = 0;
            zzss = []
            while (i < len(xs) and j < len(Ts)):
                zzs = (Ts[j] - xs[i]);
                zzss.append(zzs)
                i += 1;
                j += 1
            # print('zzss = ', zzss)
            error1 = np.linalg.norm((zzss), ord=inf, axis=None, keepdims=False)
            print("\033[1;32m Error1= \033[1;m:", error1)
            error2 = maximum.reduce(fabs(zzss))
            print("\033[1;32m Error2= \033[1;m:", error2)
            rate = log2(0.000244431588331 / 6.11172499153e-05);
            print('rate = ', rate)
            return xs, error1, error2

        # x,error = GaussSiedelIteration(A,b)
        print("***************************************************************************************")
        print("***************************************************************************************")
        print('\n\n')

        def conjugategradient(A, b):
            print('CONJUGATE GRADIENT METHOD')
            xs = scipy.sparse.linalg.cg(A, b, tol=1e-20)
            print("Solution, xs = ", xs)
            # xs1 = array(xs[0])
            # xs2 = reshape(xs1,[(E-1)*(F-1),1])
            # print("Solution, xs2 = ", xs2)
            print("")
            # error = np.dot(A, x) - b
            # error = maximum.reduce(fabs(Ts - xs[0]))
            i = 0;
            j = 0;
            zzss = []
            while (i < len(xs[0]) and j < len(Ts)):
                zzs = (Ts[j] - xs[0][i]);
                zzss.append(zzs)
                i += 1;
                j += 1
            # print('zzss = ', zzss)
            error1 = np.linalg.norm((zzss), ord=inf, axis=None, keepdims=False)
            print("\033[1;32m Error1= \033[1;m:", error1)
            error2 = maximum.reduce(fabs(zzss))
            print("\033[1;32m Error2= \033[1;m:", error2)
            rate = log2(0.00181843774534 / 0.000455832430482);
            print('rate = ', rate)
            return xs, error1, error2

        print("***************************************************************************************")
        print("***************************************************************************************")
        print('\n\n')

        if ((q, q1) == (1, 1)):
            xs,error1,error2 = QRDecomposition(A, b)
        elif ((q, q1) == (1, 2)):
            xs,error1,error2 = LUFactorization(A, b)
        elif ((q, q1) == (1, 3)):
            xs,error1,error2 = CholeskyDecomposition(A, b)
        elif ((q, q1) == (2, 1)):
            xs, error1, error2 = JacobiIteration(A, b)
        elif ((q, q1) == (2, 2)):
            xs, error1, error2 = GaussSiedelIteration(A, b)
        else:
            xs, error1, error2 = conjugategradient(A, b)
        return (D, DD,xs,error1,error2)
    D, DD,xs,error1,error2 = App_Solution(A,E,F,v,b,Ts,r,s,TS1)
    print('')
    print('')

    f = plt.figure()
    a = f.gca(projection='3d')
    from scipy.interpolate import griddata
    grid_x , grid_y = np.mgrid[-1:1:10j, -1:1:10j]
    points = [[-0.9375, -0.9375], [-0.875, -0.9375], [-0.8125, -0.9375], [-0.75, -0.9375], [-0.6875, -0.9375], [-0.625, -0.9375], [-0.5625, -0.9375], [-0.5, -0.9375], [-0.4375, -0.9375], [-0.375, -0.9375], [-0.3125, -0.9375], [-0.25, -0.9375], [-0.1875, -0.9375], [-0.125, -0.9375], [-0.0625, -0.9375], [0.0, -0.9375], [0.0625, -0.9375], [0.125, -0.9375], [0.1875, -0.9375], [0.25, -0.9375], [0.3125, -0.9375], [0.375, -0.9375], [0.4375, -0.9375], [0.5, -0.9375], [0.5625, -0.9375], [0.625, -0.9375], [0.6875, -0.9375], [0.75, -0.9375], [0.8125, -0.9375], [0.875, -0.9375], [0.9375, -0.9375], [-0.9375, -0.875], [-0.875, -0.875], [-0.8125, -0.875], [-0.75, -0.875], [-0.6875, -0.875], [-0.625, -0.875], [-0.5625, -0.875], [-0.5, -0.875], [-0.4375, -0.875], [-0.375, -0.875], [-0.3125, -0.875], [-0.25, -0.875], [-0.1875, -0.875], [-0.125, -0.875], [-0.0625, -0.875], [0.0, -0.875], [0.0625, -0.875], [0.125, -0.875], [0.1875, -0.875], [0.25, -0.875], [0.3125, -0.875], [0.375, -0.875], [0.4375, -0.875], [0.5, -0.875], [0.5625, -0.875], [0.625, -0.875], [0.6875, -0.875], [0.75, -0.875], [0.8125, -0.875], [0.875, -0.875], [0.9375, -0.875], [-0.9375, -0.8125], [-0.875, -0.8125], [-0.8125, -0.8125], [-0.75, -0.8125], [-0.6875, -0.8125], [-0.625, -0.8125], [-0.5625, -0.8125], [-0.5, -0.8125], [-0.4375, -0.8125], [-0.375, -0.8125], [-0.3125, -0.8125], [-0.25, -0.8125], [-0.1875, -0.8125], [-0.125, -0.8125], [-0.0625, -0.8125], [0.0, -0.8125], [0.0625, -0.8125], [0.125, -0.8125], [0.1875, -0.8125], [0.25, -0.8125], [0.3125, -0.8125], [0.375, -0.8125], [0.4375, -0.8125], [0.5, -0.8125], [0.5625, -0.8125], [0.625, -0.8125], [0.6875, -0.8125], [0.75, -0.8125], [0.8125, -0.8125], [0.875, -0.8125], [0.9375, -0.8125], [-0.9375, -0.75], [-0.875, -0.75], [-0.8125, -0.75], [-0.75, -0.75], [-0.6875, -0.75], [-0.625, -0.75], [-0.5625, -0.75], [-0.5, -0.75], [-0.4375, -0.75], [-0.375, -0.75], [-0.3125, -0.75], [-0.25, -0.75], [-0.1875, -0.75], [-0.125, -0.75], [-0.0625, -0.75], [0.0, -0.75], [0.0625, -0.75], [0.125, -0.75], [0.1875, -0.75], [0.25, -0.75], [0.3125, -0.75], [0.375, -0.75], [0.4375, -0.75], [0.5, -0.75], [0.5625, -0.75], [0.625, -0.75], [0.6875, -0.75], [0.75, -0.75], [0.8125, -0.75], [0.875, -0.75], [0.9375, -0.75], [-0.9375, -0.6875], [-0.875, -0.6875], [-0.8125, -0.6875], [-0.75, -0.6875], [-0.6875, -0.6875], [-0.625, -0.6875], [-0.5625, -0.6875], [-0.5, -0.6875], [-0.4375, -0.6875], [-0.375, -0.6875], [-0.3125, -0.6875], [-0.25, -0.6875], [-0.1875, -0.6875], [-0.125, -0.6875], [-0.0625, -0.6875], [0.0, -0.6875], [0.0625, -0.6875], [0.125, -0.6875], [0.1875, -0.6875], [0.25, -0.6875], [0.3125, -0.6875], [0.375, -0.6875], [0.4375, -0.6875], [0.5, -0.6875], [0.5625, -0.6875], [0.625, -0.6875], [0.6875, -0.6875], [0.75, -0.6875], [0.8125, -0.6875], [0.875, -0.6875], [0.9375, -0.6875], [-0.9375, -0.625], [-0.875, -0.625], [-0.8125, -0.625], [-0.75, -0.625], [-0.6875, -0.625], [-0.625, -0.625], [-0.5625, -0.625], [-0.5, -0.625], [-0.4375, -0.625], [-0.375, -0.625], [-0.3125, -0.625], [-0.25, -0.625], [-0.1875, -0.625], [-0.125, -0.625], [-0.0625, -0.625], [0.0, -0.625], [0.0625, -0.625], [0.125, -0.625], [0.1875, -0.625], [0.25, -0.625], [0.3125, -0.625], [0.375, -0.625], [0.4375, -0.625], [0.5, -0.625], [0.5625, -0.625], [0.625, -0.625], [0.6875, -0.625], [0.75, -0.625], [0.8125, -0.625], [0.875, -0.625], [0.9375, -0.625], [-0.9375, -0.5625], [-0.875, -0.5625], [-0.8125, -0.5625], [-0.75, -0.5625], [-0.6875, -0.5625], [-0.625, -0.5625], [-0.5625, -0.5625], [-0.5, -0.5625], [-0.4375, -0.5625], [-0.375, -0.5625], [-0.3125, -0.5625], [-0.25, -0.5625], [-0.1875, -0.5625], [-0.125, -0.5625], [-0.0625, -0.5625], [0.0, -0.5625], [0.0625, -0.5625], [0.125, -0.5625], [0.1875, -0.5625], [0.25, -0.5625], [0.3125, -0.5625], [0.375, -0.5625], [0.4375, -0.5625], [0.5, -0.5625], [0.5625, -0.5625], [0.625, -0.5625], [0.6875, -0.5625], [0.75, -0.5625], [0.8125, -0.5625], [0.875, -0.5625], [0.9375, -0.5625], [-0.9375, -0.5], [-0.875, -0.5], [-0.8125, -0.5], [-0.75, -0.5], [-0.6875, -0.5], [-0.625, -0.5], [-0.5625, -0.5], [-0.5, -0.5], [-0.4375, -0.5], [-0.375, -0.5], [-0.3125, -0.5], [-0.25, -0.5], [-0.1875, -0.5], [-0.125, -0.5], [-0.0625, -0.5], [0.0, -0.5], [0.0625, -0.5], [0.125, -0.5], [0.1875, -0.5], [0.25, -0.5], [0.3125, -0.5], [0.375, -0.5], [0.4375, -0.5], [0.5, -0.5], [0.5625, -0.5], [0.625, -0.5], [0.6875, -0.5], [0.75, -0.5], [0.8125, -0.5], [0.875, -0.5], [0.9375, -0.5], [-0.9375, -0.4375], [-0.875, -0.4375], [-0.8125, -0.4375], [-0.75, -0.4375], [-0.6875, -0.4375], [-0.625, -0.4375], [-0.5625, -0.4375], [-0.5, -0.4375], [-0.4375, -0.4375], [-0.375, -0.4375], [-0.3125, -0.4375], [-0.25, -0.4375], [-0.1875, -0.4375], [-0.125, -0.4375], [-0.0625, -0.4375], [0.0, -0.4375], [0.0625, -0.4375], [0.125, -0.4375], [0.1875, -0.4375], [0.25, -0.4375], [0.3125, -0.4375], [0.375, -0.4375], [0.4375, -0.4375], [0.5, -0.4375], [0.5625, -0.4375], [0.625, -0.4375], [0.6875, -0.4375], [0.75, -0.4375], [0.8125, -0.4375], [0.875, -0.4375], [0.9375, -0.4375], [-0.9375, -0.375], [-0.875, -0.375], [-0.8125, -0.375], [-0.75, -0.375], [-0.6875, -0.375], [-0.625, -0.375], [-0.5625, -0.375], [-0.5, -0.375], [-0.4375, -0.375], [-0.375, -0.375], [-0.3125, -0.375], [-0.25, -0.375], [-0.1875, -0.375], [-0.125, -0.375], [-0.0625, -0.375], [0.0, -0.375], [0.0625, -0.375], [0.125, -0.375], [0.1875, -0.375], [0.25, -0.375], [0.3125, -0.375], [0.375, -0.375], [0.4375, -0.375], [0.5, -0.375], [0.5625, -0.375], [0.625, -0.375], [0.6875, -0.375], [0.75, -0.375], [0.8125, -0.375], [0.875, -0.375], [0.9375, -0.375], [-0.9375, -0.3125], [-0.875, -0.3125], [-0.8125, -0.3125], [-0.75, -0.3125], [-0.6875, -0.3125], [-0.625, -0.3125], [-0.5625, -0.3125], [-0.5, -0.3125], [-0.4375, -0.3125], [-0.375, -0.3125], [-0.3125, -0.3125], [-0.25, -0.3125], [-0.1875, -0.3125], [-0.125, -0.3125], [-0.0625, -0.3125], [0.0, -0.3125], [0.0625, -0.3125], [0.125, -0.3125], [0.1875, -0.3125], [0.25, -0.3125], [0.3125, -0.3125], [0.375, -0.3125], [0.4375, -0.3125], [0.5, -0.3125], [0.5625, -0.3125], [0.625, -0.3125], [0.6875, -0.3125], [0.75, -0.3125], [0.8125, -0.3125], [0.875, -0.3125], [0.9375, -0.3125], [-0.9375, -0.25], [-0.875, -0.25], [-0.8125, -0.25], [-0.75, -0.25], [-0.6875, -0.25], [-0.625, -0.25], [-0.5625, -0.25], [-0.5, -0.25], [-0.4375, -0.25], [-0.375, -0.25], [-0.3125, -0.25], [-0.25, -0.25], [-0.1875, -0.25], [-0.125, -0.25], [-0.0625, -0.25], [0.0, -0.25], [0.0625, -0.25], [0.125, -0.25], [0.1875, -0.25], [0.25, -0.25], [0.3125, -0.25], [0.375, -0.25], [0.4375, -0.25], [0.5, -0.25], [0.5625, -0.25], [0.625, -0.25], [0.6875, -0.25], [0.75, -0.25], [0.8125, -0.25], [0.875, -0.25], [0.9375, -0.25], [-0.9375, -0.1875], [-0.875, -0.1875], [-0.8125, -0.1875], [-0.75, -0.1875], [-0.6875, -0.1875], [-0.625, -0.1875], [-0.5625, -0.1875], [-0.5, -0.1875], [-0.4375, -0.1875], [-0.375, -0.1875], [-0.3125, -0.1875], [-0.25, -0.1875], [-0.1875, -0.1875], [-0.125, -0.1875], [-0.0625, -0.1875], [0.0, -0.1875], [0.0625, -0.1875], [0.125, -0.1875], [0.1875, -0.1875], [0.25, -0.1875], [0.3125, -0.1875], [0.375, -0.1875], [0.4375, -0.1875], [0.5, -0.1875], [0.5625, -0.1875], [0.625, -0.1875], [0.6875, -0.1875], [0.75, -0.1875], [0.8125, -0.1875], [0.875, -0.1875], [0.9375, -0.1875], [-0.9375, -0.125], [-0.875, -0.125], [-0.8125, -0.125], [-0.75, -0.125], [-0.6875, -0.125], [-0.625, -0.125], [-0.5625, -0.125], [-0.5, -0.125], [-0.4375, -0.125], [-0.375, -0.125], [-0.3125, -0.125], [-0.25, -0.125], [-0.1875, -0.125], [-0.125, -0.125], [-0.0625, -0.125], [0.0, -0.125], [0.0625, -0.125], [0.125, -0.125], [0.1875, -0.125], [0.25, -0.125], [0.3125, -0.125], [0.375, -0.125], [0.4375, -0.125], [0.5, -0.125], [0.5625, -0.125], [0.625, -0.125], [0.6875, -0.125], [0.75, -0.125], [0.8125, -0.125], [0.875, -0.125], [0.9375, -0.125], [-0.9375, -0.0625], [-0.875, -0.0625], [-0.8125, -0.0625], [-0.75, -0.0625], [-0.6875, -0.0625], [-0.625, -0.0625], [-0.5625, -0.0625], [-0.5, -0.0625], [-0.4375, -0.0625], [-0.375, -0.0625], [-0.3125, -0.0625], [-0.25, -0.0625], [-0.1875, -0.0625], [-0.125, -0.0625], [-0.0625, -0.0625], [0.0, -0.0625], [0.0625, -0.0625], [0.125, -0.0625], [0.1875, -0.0625], [0.25, -0.0625], [0.3125, -0.0625], [0.375, -0.0625], [0.4375, -0.0625], [0.5, -0.0625], [0.5625, -0.0625], [0.625, -0.0625], [0.6875, -0.0625], [0.75, -0.0625], [0.8125, -0.0625], [0.875, -0.0625], [0.9375, -0.0625], [-0.9375, 0.0], [-0.875, 0.0], [-0.8125, 0.0], [-0.75, 0.0], [-0.6875, 0.0], [-0.625, 0.0], [-0.5625, 0.0], [-0.5, 0.0], [-0.4375, 0.0], [-0.375, 0.0], [-0.3125, 0.0], [-0.25, 0.0], [-0.1875, 0.0], [-0.125, 0.0], [-0.0625, 0.0], [0.0, 0.0], [0.0625, 0.0], [0.125, 0.0], [0.1875, 0.0], [0.25, 0.0], [0.3125, 0.0], [0.375, 0.0], [0.4375, 0.0], [0.5, 0.0], [0.5625, 0.0], [0.625, 0.0], [0.6875, 0.0], [0.75, 0.0], [0.8125, 0.0], [0.875, 0.0], [0.9375, 0.0], [-0.9375, 0.0625], [-0.875, 0.0625], [-0.8125, 0.0625], [-0.75, 0.0625], [-0.6875, 0.0625], [-0.625, 0.0625], [-0.5625, 0.0625], [-0.5, 0.0625], [-0.4375, 0.0625], [-0.375, 0.0625], [-0.3125, 0.0625], [-0.25, 0.0625], [-0.1875, 0.0625], [-0.125, 0.0625], [-0.0625, 0.0625], [0.0, 0.0625], [0.0625, 0.0625], [0.125, 0.0625], [0.1875, 0.0625], [0.25, 0.0625], [0.3125, 0.0625], [0.375, 0.0625], [0.4375, 0.0625], [0.5, 0.0625], [0.5625, 0.0625], [0.625, 0.0625], [0.6875, 0.0625], [0.75, 0.0625], [0.8125, 0.0625], [0.875, 0.0625], [0.9375, 0.0625], [-0.9375, 0.125], [-0.875, 0.125], [-0.8125, 0.125], [-0.75, 0.125], [-0.6875, 0.125], [-0.625, 0.125], [-0.5625, 0.125], [-0.5, 0.125], [-0.4375, 0.125], [-0.375, 0.125], [-0.3125, 0.125], [-0.25, 0.125], [-0.1875, 0.125], [-0.125, 0.125], [-0.0625, 0.125], [0.0, 0.125], [0.0625, 0.125], [0.125, 0.125], [0.1875, 0.125], [0.25, 0.125], [0.3125, 0.125], [0.375, 0.125], [0.4375, 0.125], [0.5, 0.125], [0.5625, 0.125], [0.625, 0.125], [0.6875, 0.125], [0.75, 0.125], [0.8125, 0.125], [0.875, 0.125], [0.9375, 0.125], [-0.9375, 0.1875], [-0.875, 0.1875], [-0.8125, 0.1875], [-0.75, 0.1875], [-0.6875, 0.1875], [-0.625, 0.1875], [-0.5625, 0.1875], [-0.5, 0.1875], [-0.4375, 0.1875], [-0.375, 0.1875], [-0.3125, 0.1875], [-0.25, 0.1875], [-0.1875, 0.1875], [-0.125, 0.1875], [-0.0625, 0.1875], [0.0, 0.1875], [0.0625, 0.1875], [0.125, 0.1875], [0.1875, 0.1875], [0.25, 0.1875], [0.3125, 0.1875], [0.375, 0.1875], [0.4375, 0.1875], [0.5, 0.1875], [0.5625, 0.1875], [0.625, 0.1875], [0.6875, 0.1875], [0.75, 0.1875], [0.8125, 0.1875], [0.875, 0.1875], [0.9375, 0.1875], [-0.9375, 0.25], [-0.875, 0.25], [-0.8125, 0.25], [-0.75, 0.25], [-0.6875, 0.25], [-0.625, 0.25], [-0.5625, 0.25], [-0.5, 0.25], [-0.4375, 0.25], [-0.375, 0.25], [-0.3125, 0.25], [-0.25, 0.25], [-0.1875, 0.25], [-0.125, 0.25], [-0.0625, 0.25], [0.0, 0.25], [0.0625, 0.25], [0.125, 0.25], [0.1875, 0.25], [0.25, 0.25], [0.3125, 0.25], [0.375, 0.25], [0.4375, 0.25], [0.5, 0.25], [0.5625, 0.25], [0.625, 0.25], [0.6875, 0.25], [0.75, 0.25], [0.8125, 0.25], [0.875, 0.25], [0.9375, 0.25], [-0.9375, 0.3125], [-0.875, 0.3125], [-0.8125, 0.3125], [-0.75, 0.3125], [-0.6875, 0.3125], [-0.625, 0.3125], [-0.5625, 0.3125], [-0.5, 0.3125], [-0.4375, 0.3125], [-0.375, 0.3125], [-0.3125, 0.3125], [-0.25, 0.3125], [-0.1875, 0.3125], [-0.125, 0.3125], [-0.0625, 0.3125], [0.0, 0.3125], [0.0625, 0.3125], [0.125, 0.3125], [0.1875, 0.3125], [0.25, 0.3125], [0.3125, 0.3125], [0.375, 0.3125], [0.4375, 0.3125], [0.5, 0.3125], [0.5625, 0.3125], [0.625, 0.3125], [0.6875, 0.3125], [0.75, 0.3125], [0.8125, 0.3125], [0.875, 0.3125], [0.9375, 0.3125], [-0.9375, 0.375], [-0.875, 0.375], [-0.8125, 0.375], [-0.75, 0.375], [-0.6875, 0.375], [-0.625, 0.375], [-0.5625, 0.375], [-0.5, 0.375], [-0.4375, 0.375], [-0.375, 0.375], [-0.3125, 0.375], [-0.25, 0.375], [-0.1875, 0.375], [-0.125, 0.375], [-0.0625, 0.375], [0.0, 0.375], [0.0625, 0.375], [0.125, 0.375], [0.1875, 0.375], [0.25, 0.375], [0.3125, 0.375], [0.375, 0.375], [0.4375, 0.375], [0.5, 0.375], [0.5625, 0.375], [0.625, 0.375], [0.6875, 0.375], [0.75, 0.375], [0.8125, 0.375], [0.875, 0.375], [0.9375, 0.375], [-0.9375, 0.4375], [-0.875, 0.4375], [-0.8125, 0.4375], [-0.75, 0.4375], [-0.6875, 0.4375], [-0.625, 0.4375], [-0.5625, 0.4375], [-0.5, 0.4375], [-0.4375, 0.4375], [-0.375, 0.4375], [-0.3125, 0.4375], [-0.25, 0.4375], [-0.1875, 0.4375], [-0.125, 0.4375], [-0.0625, 0.4375], [0.0, 0.4375], [0.0625, 0.4375], [0.125, 0.4375], [0.1875, 0.4375], [0.25, 0.4375], [0.3125, 0.4375], [0.375, 0.4375], [0.4375, 0.4375], [0.5, 0.4375], [0.5625, 0.4375], [0.625, 0.4375], [0.6875, 0.4375], [0.75, 0.4375], [0.8125, 0.4375], [0.875, 0.4375], [0.9375, 0.4375], [-0.9375, 0.5], [-0.875, 0.5], [-0.8125, 0.5], [-0.75, 0.5], [-0.6875, 0.5], [-0.625, 0.5], [-0.5625, 0.5], [-0.5, 0.5], [-0.4375, 0.5], [-0.375, 0.5], [-0.3125, 0.5], [-0.25, 0.5], [-0.1875, 0.5], [-0.125, 0.5], [-0.0625, 0.5], [0.0, 0.5], [0.0625, 0.5], [0.125, 0.5], [0.1875, 0.5], [0.25, 0.5], [0.3125, 0.5], [0.375, 0.5], [0.4375, 0.5], [0.5, 0.5], [0.5625, 0.5], [0.625, 0.5], [0.6875, 0.5], [0.75, 0.5], [0.8125, 0.5], [0.875, 0.5], [0.9375, 0.5], [-0.9375, 0.5625], [-0.875, 0.5625], [-0.8125, 0.5625], [-0.75, 0.5625], [-0.6875, 0.5625], [-0.625, 0.5625], [-0.5625, 0.5625], [-0.5, 0.5625], [-0.4375, 0.5625], [-0.375, 0.5625], [-0.3125, 0.5625], [-0.25, 0.5625], [-0.1875, 0.5625], [-0.125, 0.5625], [-0.0625, 0.5625], [0.0, 0.5625], [0.0625, 0.5625], [0.125, 0.5625], [0.1875, 0.5625], [0.25, 0.5625], [0.3125, 0.5625], [0.375, 0.5625], [0.4375, 0.5625], [0.5, 0.5625], [0.5625, 0.5625], [0.625, 0.5625], [0.6875, 0.5625], [0.75, 0.5625], [0.8125, 0.5625], [0.875, 0.5625], [0.9375, 0.5625], [-0.9375, 0.625], [-0.875, 0.625], [-0.8125, 0.625], [-0.75, 0.625], [-0.6875, 0.625], [-0.625, 0.625], [-0.5625, 0.625], [-0.5, 0.625], [-0.4375, 0.625], [-0.375, 0.625], [-0.3125, 0.625], [-0.25, 0.625], [-0.1875, 0.625], [-0.125, 0.625], [-0.0625, 0.625], [0.0, 0.625], [0.0625, 0.625], [0.125, 0.625], [0.1875, 0.625], [0.25, 0.625], [0.3125, 0.625], [0.375, 0.625], [0.4375, 0.625], [0.5, 0.625], [0.5625, 0.625], [0.625, 0.625], [0.6875, 0.625], [0.75, 0.625], [0.8125, 0.625], [0.875, 0.625], [0.9375, 0.625], [-0.9375, 0.6875], [-0.875, 0.6875], [-0.8125, 0.6875], [-0.75, 0.6875], [-0.6875, 0.6875], [-0.625, 0.6875], [-0.5625, 0.6875], [-0.5, 0.6875], [-0.4375, 0.6875], [-0.375, 0.6875], [-0.3125, 0.6875], [-0.25, 0.6875], [-0.1875, 0.6875], [-0.125, 0.6875], [-0.0625, 0.6875], [0.0, 0.6875], [0.0625, 0.6875], [0.125, 0.6875], [0.1875, 0.6875], [0.25, 0.6875], [0.3125, 0.6875], [0.375, 0.6875], [0.4375, 0.6875], [0.5, 0.6875], [0.5625, 0.6875], [0.625, 0.6875], [0.6875, 0.6875], [0.75, 0.6875], [0.8125, 0.6875], [0.875, 0.6875], [0.9375, 0.6875], [-0.9375, 0.75], [-0.875, 0.75], [-0.8125, 0.75], [-0.75, 0.75], [-0.6875, 0.75], [-0.625, 0.75], [-0.5625, 0.75], [-0.5, 0.75], [-0.4375, 0.75], [-0.375, 0.75], [-0.3125, 0.75], [-0.25, 0.75], [-0.1875, 0.75], [-0.125, 0.75], [-0.0625, 0.75], [0.0, 0.75], [0.0625, 0.75], [0.125, 0.75], [0.1875, 0.75], [0.25, 0.75], [0.3125, 0.75], [0.375, 0.75], [0.4375, 0.75], [0.5, 0.75], [0.5625, 0.75], [0.625, 0.75], [0.6875, 0.75], [0.75, 0.75], [0.8125, 0.75], [0.875, 0.75], [0.9375, 0.75], [-0.9375, 0.8125], [-0.875, 0.8125], [-0.8125, 0.8125], [-0.75, 0.8125], [-0.6875, 0.8125], [-0.625, 0.8125], [-0.5625, 0.8125], [-0.5, 0.8125], [-0.4375, 0.8125], [-0.375, 0.8125], [-0.3125, 0.8125], [-0.25, 0.8125], [-0.1875, 0.8125], [-0.125, 0.8125], [-0.0625, 0.8125], [0.0, 0.8125], [0.0625, 0.8125], [0.125, 0.8125], [0.1875, 0.8125], [0.25, 0.8125], [0.3125, 0.8125], [0.375, 0.8125], [0.4375, 0.8125], [0.5, 0.8125], [0.5625, 0.8125], [0.625, 0.8125], [0.6875, 0.8125], [0.75, 0.8125], [0.8125, 0.8125], [0.875, 0.8125], [0.9375, 0.8125], [-0.9375, 0.875], [-0.875, 0.875], [-0.8125, 0.875], [-0.75, 0.875], [-0.6875, 0.875], [-0.625, 0.875], [-0.5625, 0.875], [-0.5, 0.875], [-0.4375, 0.875], [-0.375, 0.875], [-0.3125, 0.875], [-0.25, 0.875], [-0.1875, 0.875], [-0.125, 0.875], [-0.0625, 0.875], [0.0, 0.875], [0.0625, 0.875], [0.125, 0.875], [0.1875, 0.875], [0.25, 0.875], [0.3125, 0.875], [0.375, 0.875], [0.4375, 0.875], [0.5, 0.875], [0.5625, 0.875], [0.625, 0.875], [0.6875, 0.875], [0.75, 0.875], [0.8125, 0.875], [0.875, 0.875], [0.9375, 0.875], [-0.9375, 0.9375], [-0.875, 0.9375], [-0.8125, 0.9375], [-0.75, 0.9375], [-0.6875, 0.9375], [-0.625, 0.9375], [-0.5625, 0.9375], [-0.5, 0.9375], [-0.4375, 0.9375], [-0.375, 0.9375], [-0.3125, 0.9375], [-0.25, 0.9375], [-0.1875, 0.9375], [-0.125, 0.9375], [-0.0625, 0.9375], [0.0, 0.9375], [0.0625, 0.9375], [0.125, 0.9375], [0.1875, 0.9375], [0.25, 0.9375], [0.3125, 0.9375], [0.375, 0.9375], [0.4375, 0.9375], [0.5, 0.9375], [0.5625, 0.9375], [0.625, 0.9375], [0.6875, 0.9375], [0.75, 0.9375], [0.8125, 0.9375], [0.875, 0.9375], [0.9375, 0.9375]]



    values = [ 0.03529098,  0.06442   ,  0.08809619,  0.1069608 ,  0.12159311,
        0.13251586,  0.14020017,  0.14507009,  0.14750675,  0.14785216,
        0.14641273,  0.14346247,  0.13924591,  0.13398082,  0.12786068,
        0.12105693,  0.11372105,  0.10598645,  0.09797022,  0.0897747 ,
        0.08148894,  0.07319   ,  0.06494421,  0.0568082 ,  0.04882998,
        0.04104978,  0.03350094,  0.02621064,  0.01920061,  0.01248771,
        0.00608455,  0.06442   ,  0.11805152,  0.16206979,  0.19754405,
        0.22544621,  0.24665887,  0.26198267,  0.27214306,  0.27779654,
        0.27953639,  0.27789802,  0.27336379,  0.26636758,  0.25729891,
        0.24650677,  0.23430318,  0.22096639,  0.20674389,  0.19185518,
        0.17649429,  0.16083209,  0.14501848,  0.12918427,  0.11344308,
        0.09789293,  0.08261779,  0.06768896,  0.05316633,  0.03909959,
        0.02552926,  0.01248771,  0.08809619,  0.16206979,  0.22337105,
        0.27332749,  0.31315335,  0.34395834,  0.36675569,  0.38246962,
        0.39194224,  0.39593995,  0.39515939,  0.39023291,  0.38173367,
        0.37018037,  0.35604163,  0.33974004,  0.32165589,  0.3021307 ,
        0.28147039,  0.2599483 ,  0.23780791,  0.21526544,  0.19251213,
        0.16971651,  0.14702634,  0.12457051,  0.10246072,  0.0807931 ,
        0.05964965,  0.03909959,  0.01920061,  0.1069608 ,  0.19754405,
        0.27332749,  0.33576425,  0.38619185,  0.42584051,  0.45584086,
        0.47723112,  0.4909638 ,  0.49791186,  0.49887455,  0.49458275,
        0.48570404,  0.47284733,  0.45656727,  0.43736828,  0.41570835,
        0.39200256,  0.36662637,  0.33991866,  0.31218461,  0.28369833,
        0.25470529,  0.22542468,  0.19605147,  0.16675847,  0.13769807,
        0.10900405,  0.0807931 ,  0.05316633,  0.02621064,  0.12159311,
        0.22544621,  0.31315335,  0.38619185,  0.44593012,  0.49363491,
        0.53047814,  0.55754316,  0.57583073,  0.58626453,  0.5896963 ,
        0.58691075,  0.57863004,  0.56551807,  0.54818448,  0.5271883 ,
        0.50304156,  0.47621248,  0.44712858,  0.41617953,  0.38371986,
        0.35007149,  0.31552605,  0.28034712,  0.24477229,  0.20901509,
        0.17326677,  0.13769807,  0.10246072,  0.06768896,  0.03350094,
        0.13251586,  0.24665887,  0.34395834,  0.42584051,  0.49363491,
        0.54858035,  0.59183046,  0.62445894,  0.64746446,  0.66177526,
        0.66825352,  0.66769937,  0.6608548 ,  0.64840723,  0.63099291,
        0.60920016,  0.58357236,  0.5546108 ,  0.52277738,  0.48849708,
        0.45216043,  0.41412564,  0.3747208 ,  0.33424581,  0.2929743 ,
        0.25115532,  0.20901509,  0.16675847,  0.12457051,  0.08261779,
        0.04104978,  0.14020017,  0.26198267,  0.36675569,  0.45584086,
        0.53047814,  0.59183046,  0.64098804,  0.67897245,  0.70674048,
        0.72518771,  0.73515199,  0.73741664,  0.73271351,  0.72172588,
        0.70509119,  0.68340366,  0.65721668,  0.62704521,  0.59336793,
        0.55662936,  0.51724182,  0.47558727,  0.43201918,  0.38686408,
        0.34042329,  0.2929743 ,  0.24477229,  0.19605147,  0.14702634,
        0.09789293,  0.04882998,  0.14507009,  0.27214306,  0.38246962,
        0.47723112,  0.55754316,  0.62445894,  0.67897245,  0.7220215 ,
        0.7544905 ,  0.77721322,  0.79097526,  0.79651656,  0.79453362,
        0.78568173,  0.77057701,  0.74979842,  0.72388962,  0.69336076,
        0.65869019,  0.62032607,  0.57868794,  0.53416815,  0.48713332,
        0.43792562,  0.38686408,  0.33424581,  0.28034712,  0.22542468,
        0.16971651,  0.11344308,  0.0568082 ,  0.14750675,  0.27779654,
        0.39194224,  0.4909638 ,  0.57583073,  0.64746446,  0.70674048,
        0.7544905 ,  0.79150441,  0.81853213,  0.83628542,  0.8454396 ,
        0.84663513,  0.84047919,  0.82754714,  0.80838393,  0.78350548,
        0.75339989,  0.71852879,  0.67932839,  0.63621073,  0.58956467,
        0.53975698,  0.48713332,  0.43201918,  0.3747208 ,  0.31552605,
        0.25470529,  0.19251213,  0.12918427,  0.06494421,  0.14785216,
        0.27953639,  0.39593995,  0.49791186,  0.58626453,  0.66177526,
        0.72518771,  0.77721322,  0.81853213,  0.84979502,  0.87162384,
        0.88461307,  0.88933074,  0.88631951,  0.87609757,  0.85915963,
        0.83597781,  0.80700245,  0.772663  ,  0.73336879,  0.68950977,
        0.64145726,  0.58956467,  0.53416815,  0.47558727,  0.41412564,
        0.35007149,  0.28369833,  0.21526544,  0.14501848,  0.07319   ,
        0.14641273,  0.27789802,  0.39515939,  0.49887455,  0.5896963 ,
        0.66825352,  0.73515199,  0.79097526,  0.83628542,  0.87162384,
        0.89751186,  0.91445151,  0.9229261 ,  0.92340086,  0.91632353,
        0.90212493,  0.88121946,  0.85400565,  0.82086668,  0.78217082,
        0.7382719 ,  0.68950977,  0.63621073,  0.57868794,  0.51724182,
        0.45216043,  0.38371986,  0.31218461,  0.23780791,  0.16083209,
        0.08148894,  0.14346247,  0.27336379,  0.39023291,  0.49458275,
        0.58691075,  0.66769937,  0.73741664,  0.79651656,  0.8454396 ,
        0.88461307,  0.91445151,  0.93535709,  0.94771992,  0.9519184 ,
        0.94831955,  0.93727925,  0.9191426 ,  0.89424416,  0.86290818,
        0.82544891,  0.78217082,  0.73336879,  0.67932839,  0.62032607,
        0.55662936,  0.48849708,  0.41617953,  0.33991866,  0.2599483 ,
        0.17649429,  0.0897747 ,  0.13924591,  0.26636758,  0.38173367,
        0.48570404,  0.57863004,  0.6608548 ,  0.73271351,  0.79453362,
        0.84663513,  0.88933074,  0.9229261 ,  0.94771992,  0.96400421,
        0.9720644 ,  0.97217946,  0.96462208,  0.94965878,  0.92755   ,
        0.89855025,  0.86290818,  0.82086668,  0.772663  ,  0.71852879,
        0.65869019,  0.59336793,  0.52277738,  0.44712858,  0.36662637,
        0.28147039,  0.19185518,  0.09797022,  0.13398082,  0.25729891,
        0.37018037,  0.47284733,  0.56551807,  0.64840723,  0.72172588,
        0.78568173,  0.84047919,  0.88631951,  0.92340086,  0.9519184 ,
        0.9720644 ,  0.98402822,  0.98799648,  0.984153  ,  0.9726789 ,
        0.95375264,  0.92755   ,  0.89424416,  0.85400565,  0.80700245,
        0.75339989,  0.69336076,  0.62704521,  0.5546108 ,  0.47621248,
        0.39200256,  0.3021307 ,  0.20674389,  0.10598645,  0.12786068,
        0.24650677,  0.35604163,  0.45656727,  0.54818448,  0.63099291,
        0.70509119,  0.77057701,  0.82754714,  0.87609757,  0.91632353,
        0.94831955,  0.97217946,  0.98799648,  0.99586322,  0.99587168,
        0.98811329,  0.9726789 ,  0.94965878,  0.9191426 ,  0.88121946,
        0.83597781,  0.78350548,  0.72388962,  0.65721668,  0.58357236,
        0.50304156,  0.41570835,  0.32165589,  0.22096639,  0.11372105,
        0.12105693,  0.23430318,  0.33974004,  0.43736828,  0.5271883 ,
        0.60920016,  0.68340366,  0.74979842,  0.80838393,  0.85915963,
        0.90212493,  0.93727925,  0.96462208,  0.984153  ,  0.99587168,
        0.99977793,  0.99587168,  0.984153  ,  0.96462208,  0.93727925,
        0.90212493,  0.85915963,  0.80838393,  0.74979842,  0.68340366,
        0.60920016,  0.5271883 ,  0.43736828,  0.33974004,  0.23430318,
        0.12105693,  0.11372105,  0.22096639,  0.32165589,  0.41570835,
        0.50304156,  0.58357236,  0.65721668,  0.72388962,  0.78350548,
        0.83597781,  0.88121946,  0.9191426 ,  0.94965878,  0.9726789 ,
        0.98811329,  0.99587168,  0.99586322,  0.98799648,  0.97217946,
        0.94831955,  0.91632353,  0.87609757,  0.82754714,  0.77057701,
        0.70509119,  0.63099291,  0.54818448,  0.45656727,  0.35604163,
        0.24650677,  0.12786068,  0.10598645,  0.20674389,  0.3021307 ,
        0.39200256,  0.47621248,  0.5546108 ,  0.62704521,  0.69336076,
        0.75339989,  0.80700245,  0.85400565,  0.89424416,  0.92755   ,
        0.95375264,  0.9726789 ,  0.984153  ,  0.98799648,  0.98402822,
        0.9720644 ,  0.9519184 ,  0.92340086,  0.88631951,  0.84047919,
        0.78568173,  0.72172588,  0.64840723,  0.56551807,  0.47284733,
        0.37018037,  0.25729891,  0.13398082,  0.09797022,  0.19185518,
        0.28147039,  0.36662637,  0.44712858,  0.52277738,  0.59336793,
        0.65869019,  0.71852879,  0.772663  ,  0.82086668,  0.86290818,
        0.89855025,  0.92755   ,  0.94965878,  0.96462208,  0.97217946,
        0.9720644 ,  0.96400421,  0.94771992,  0.9229261 ,  0.88933074,
        0.84663513,  0.79453362,  0.73271351,  0.6608548 ,  0.57863004,
        0.48570404,  0.38173367,  0.26636758,  0.13924591,  0.0897747 ,
        0.17649429,  0.2599483 ,  0.33991866,  0.41617953,  0.48849708,
        0.55662936,  0.62032607,  0.67932839,  0.73336879,  0.78217082,
        0.82544891,  0.86290818,  0.89424416,  0.9191426 ,  0.93727925,
        0.94831955,  0.9519184 ,  0.94771992,  0.93535709,  0.91445151,
        0.88461307,  0.8454396 ,  0.79651656,  0.73741664,  0.66769937,
        0.58691075,  0.49458275,  0.39023291,  0.27336379,  0.14346247,
        0.08148894,  0.16083209,  0.23780791,  0.31218461,  0.38371986,
        0.45216043,  0.51724182,  0.57868794,  0.63621073,  0.68950977,
        0.7382719 ,  0.78217082,  0.82086668,  0.85400565,  0.88121946,
        0.90212493,  0.91632353,  0.92340086,  0.9229261 ,  0.91445151,
        0.89751186,  0.87162384,  0.83628542,  0.79097526,  0.73515199,
        0.66825352,  0.5896963 ,  0.49887455,  0.39515939,  0.27789802,
        0.14641273,  0.07319   ,  0.14501848,  0.21526544,  0.28369833,
        0.35007149,  0.41412564,  0.47558727,  0.53416815,  0.58956467,
        0.64145726,  0.68950977,  0.73336879,  0.772663  ,  0.80700245,
        0.83597781,  0.85915963,  0.87609757,  0.88631951,  0.88933074,
        0.88461307,  0.87162384,  0.84979502,  0.81853213,  0.77721322,
        0.72518771,  0.66177526,  0.58626453,  0.49791186,  0.39593995,
        0.27953639,  0.14785216,  0.06494421,  0.12918427,  0.19251213,
        0.25470529,  0.31552605,  0.3747208 ,  0.43201918,  0.48713332,
        0.53975698,  0.58956467,  0.63621073,  0.67932839,  0.71852879,
        0.75339989,  0.78350548,  0.80838393,  0.82754714,  0.84047919,
        0.84663513,  0.8454396 ,  0.83628542,  0.81853213,  0.79150441,
        0.7544905 ,  0.70674048,  0.64746446,  0.57583073,  0.4909638 ,
        0.39194224,  0.27779654,  0.14750675,  0.0568082 ,  0.11344308,
        0.16971651,  0.22542468,  0.28034712,  0.33424581,  0.38686408,
        0.43792562,  0.48713332,  0.53416815,  0.57868794,  0.62032607,
        0.65869019,  0.69336076,  0.72388962,  0.74979842,  0.77057701,
        0.78568173,  0.79453362,  0.79651656,  0.79097526,  0.77721322,
        0.7544905 ,  0.7220215 ,  0.67897245,  0.62445894,  0.55754316,
        0.47723112,  0.38246962,  0.27214306,  0.14507009,  0.04882998,
        0.09789293,  0.14702634,  0.19605147,  0.24477229,  0.2929743 ,
        0.34042329,  0.38686408,  0.43201918,  0.47558727,  0.51724182,
        0.55662936,  0.59336793,  0.62704521,  0.65721668,  0.68340366,
        0.70509119,  0.72172588,  0.73271351,  0.73741664,  0.73515199,
        0.72518771,  0.70674048,  0.67897245,  0.64098804,  0.59183046,
        0.53047814,  0.45584086,  0.36675569,  0.26198267,  0.14020017,
        0.04104978,  0.08261779,  0.12457051,  0.16675847,  0.20901509,
        0.25115532,  0.2929743 ,  0.33424581,  0.3747208 ,  0.41412564,
        0.45216043,  0.48849708,  0.52277738,  0.5546108 ,  0.58357236,
        0.60920016,  0.63099291,  0.64840723,  0.6608548 ,  0.66769937,
        0.66825352,  0.66177526,  0.64746446,  0.62445894,  0.59183046,
        0.54858035,  0.49363491,  0.42584051,  0.34395834,  0.24665887,
        0.13251586,  0.03350094,  0.06768896,  0.10246072,  0.13769807,
        0.17326677,  0.20901509,  0.24477229,  0.28034712,  0.31552605,
        0.35007149,  0.38371986,  0.41617953,  0.44712858,  0.47621248,
        0.50304156,  0.5271883 ,  0.54818448,  0.56551807,  0.57863004,
        0.58691075,  0.5896963 ,  0.58626453,  0.57583073,  0.55754316,
        0.53047814,  0.49363491,  0.44593012,  0.38619185,  0.31315335,
        0.22544621,  0.12159311,  0.02621064,  0.05316633,  0.0807931 ,
        0.10900405,  0.13769807,  0.16675847,  0.19605147,  0.22542468,
        0.25470529,  0.28369833,  0.31218461,  0.33991866,  0.36662637,
        0.39200256,  0.41570835,  0.43736828,  0.45656727,  0.47284733,
        0.48570404,  0.49458275,  0.49887455,  0.49791186,  0.4909638 ,
        0.47723112,  0.45584086,  0.42584051,  0.38619185,  0.33576425,
        0.27332749,  0.19754405,  0.1069608 ,  0.01920061,  0.03909959,
        0.05964965,  0.0807931 ,  0.10246072,  0.12457051,  0.14702634,
        0.16971651,  0.19251213,  0.21526544,  0.23780791,  0.2599483 ,
        0.28147039,  0.3021307 ,  0.32165589,  0.33974004,  0.35604163,
        0.37018037,  0.38173367,  0.39023291,  0.39515939,  0.39593995,
        0.39194224,  0.38246962,  0.36675569,  0.34395834,  0.31315335,
        0.27332749,  0.22337105,  0.16206979,  0.08809619,  0.01248771,
        0.02552926,  0.03909959,  0.05316633,  0.06768896,  0.08261779,
        0.09789293,  0.11344308,  0.12918427,  0.14501848,  0.16083209,
        0.17649429,  0.19185518,  0.20674389,  0.22096639,  0.23430318,
        0.24650677,  0.25729891,  0.26636758,  0.27336379,  0.27789802,
        0.27953639,  0.27779654,  0.27214306,  0.26198267,  0.24665887,
        0.22544621,  0.19754405,  0.16206979,  0.11805152,  0.06442   ,
        0.00608455,  0.01248771,  0.01920061,  0.02621064,  0.03350094,
        0.04104978,  0.04882998,  0.0568082 ,  0.06494421,  0.07319   ,
        0.08148894,  0.0897747 ,  0.09797022,  0.10598645,  0.11372105,
        0.12105693,  0.12786068,  0.13398082,  0.13924591,  0.14346247,
        0.14641273,  0.14785216,  0.14750675,  0.14507009,  0.14020017,
        0.13251586,  0.12159311,  0.1069608 ,  0.08809619,  0.06442   ,
        0.03529098]
    grid_z0 = griddata(points, values, (grid_x,grid_y), method='linear')
   # print('grid_x,grid_y=',points)
    s=a.plot_surface(grid_x,grid_y,grid_z0)
    plt.show()


    hlist = [-log2(0.5), -log2(0.25), -log2(0.125), -log2(0.0625), -log2(0.03125)]
    error51 = [log2(0.0147006314552), log2(0.00387406969501), log2(0.000971298868515),log2(0.000244431588331), log2(6.1117249915*10**-5)]
    error52 = [log2(0.127173244643), log2(0.0296834523051), log2(0.00730252810185), log2(0.00181843774534), log2(0.000455832430482)]
    plt.plot(hlist, error52, label="inf_morm");
    plt.xlabel("log h");
    plt.ylabel("log2 (Discrete error)");
    plt.legend(['inf_norm'])
    plt.title('This is a plot of the error for the 5-point FD', color='red');
    plt.xlabel('-log2 (h)', color='red');
    plt.ylabel('log2(error)', color='red')
    plt.savefig('error52.png', transparent=True);
    plt.show()

    '''
    def vecsol(E,F,xs):
        G = zeros([(E + 1) * (F + 1), 1])
        G1 = array(G)
        if E==F==4:
            i = 0; j = E+2
            while(j<((E+2)+(F-1)) and i<(E-1)):
                a = xs[(i)*(1)]
                G1[j] = a
                i+=1;j+=1

            i = (E-1); j = (E+2)+(F+1)
            while (j<((E*F)-2) and i<(E+2)):
                a = xs[(i) * (1)]
                G1[j] = a
                i+=1; j+=1
            i = (E+2);j = E*F
            while (j < ((E*F)+F) and i < (E + (F+1))):
                a = xs[(i) * (1)]
                G1[j] = a
                i += 1;j += 1
            #print('G1 = ', G1)

        elif E>4:
            i = 0; i1 = 0
            while(i1<(E-1)):
                j = ((E+2)+(i1*(E+1)))
                while(i<((E-1)*(F-1)) and j < ((2*E)+1)+(i1*(E+1))):
                    a = xs[(i) * (1)]
                    G1[j] = a
                    i += 1; j+= 1
                i1 += 1
            #print('G1 = ', G1)
        return G1
    G1 = vecsol(E,F,xs)
    '''
    '''
    def Squares(E, F, h, k, r, s):
        z = sqrt(pow(h, 2) + pow(k, 2));d = E + 1;D = E + 2;l = len(r) - d;x10 = [];y10 = [];V = [];V1=[];nodes = [];nummer=[];address=[]
        for i in range(0, l - 1):
            for j in range(i + 1, i + 2):
                if (r[j] - r[i] <= h and s[i + d] - s[i] <= k and sqrt((r[i] - r[i + D]) ** 2 + (s[i] - s[i + D]) ** 2) <= z):
                    x = (r[i], r[j], r[i + D], r[i + d]);V1.append(x);x10.append(x)
                    y = (s[i], s[j], s[i + D], s[i + d]);V1.append(y);y10.append(y)
                    V.append([x, y]);#V.append([x, y])
        #print('V = ', V); print('V1 = ', V1); print('x10 = ', x10);print('y10 = ', y10)
        print('')

        for d in range(0, len(r)):
            nodes.append([r[d], s[d]])
        #print('nodes = ', nodes)
        for d1 in range(0, len(nodes)):
            nummer.append(nodes.index(nodes[d1]))
        #print('nummer = ', nummer)
        print('')
        a = arange(1,(E+1)*(F+1)+1, 1);#print('a = ', a)
        ii = 0; w=[]
        while(ii<(E*F)+(E-1)):
            i=0;j=0;k=0;q=2
            while(i<q and j<q and k<q):
                w.append(a[ii+(k*i)])
                i += 1;j += 1;k += 1
            i1 = 0;j1 = 0;k1 = 0;q3 = 2;q1=5
            while (i1 < q3 and j1 < q3 and k1 < q3):
                w.append(a[(ii+k1*i1)+q1])
                i1+=1;j1+=1;k1+=1
            ii+=1
        #print('w = ', w)
        return (V, x10, y10,nodes)
    V, x10, y10,nodes = Squares(E, F, h, k, r, s)
    print('')

    #approx solution
    '''
    '''
    f = plt.figure()
    a = f.gca(projection='3d')
    x = np.arange(-1, 1, 0.01)
    y = np.arange(-1, 1, 0.01)
    X, Z = np.meshgrid(x, y)
    u = (x[0] ** 2-1)**2*(y[0]**2 - 1)**2*sin(x[0])*cos(y[0])
    s = a.plot_surface(X, Z, u, cmap=cm.jet)
    f.colorbar(s, shrink=0.5);
    plt.savefig('True_solution2.png', transparent=True)
    plt.title('This is a plot of the true solution2', color='red')
    plt.xlabel('x-axis', color='red');
    plt.ylabel('y-axis', color='red')
    plt.show()
    '''
    '''
    def LagPol():
        xi, eta = symbols('xi eta')
        L1 = (1 / 4) * ((1 + xi) * (1 + eta))
        L2 = (1 / 4) * ((1 - xi) * (1 + eta))
        L3 = (1 / 4) * ((1 - xi) * (1 - eta))
        L4 = (1 / 4) * ((1 + xi) * (1 - eta))
        print('L1 = ', L1, 'L2 = ', L2, 'L3 = ', L3, 'L4 = ', L4 )
        return (L1, L2, L3, L4)
    L1, L2, L3, L4 = LagPol()
    print('')

    def trans(L1, L2, L3, L4, V):
        T = [];Z1 = [];Z2 = [];
        for j in range(0, (E + 1) * (F + 1)):
            i = 0
            T1 = L1 * V[0][0][i] + L2 * V[0][0][i + 1] + L3 * V[0][0][i + 2] + L4 * V[0][0][i + 3];
            Z1.append(T1)
            T2 = L1 * V[0][1][i] + L2 * V[0][1][i + 1] + L3 * V[0][1][i + 2] + L4 * V[0][1][i + 3];
            Z2.append(T2);
            T.append((Z1, Z2))
            # print('Z1 = ', Z1); print('Z2 = ', Z2)
        # print('T = ', T)
        return (T1, T2)
    T1, T2 = trans(L1, L2, L3, L4, V)
'''

    print('***************************************PROGRAM TERMINATED************************************************')

    def timestamp ( ):
        import time
        t = time.time ( )
        print ( time.ctime ( t ) )
        return None

    def timestamp_test ( ):
        import platform
        print ( '' )
        print ( 'TIMESTAMP_TEST:' )
        print ( '  Python version: %s' % ( platform.python_version ( ) ) )
        print ( '  TIMESTAMP prints a timestamp of the current date and time.' )
        print ( '' )

        timestamp ( )
    #
    #  Terminate.
    #
        print ( '' )
        print ( 'TIMESTAMP_TEST:' )
        print ( '  Normal end of execution.' )
        return

    if ( __name__ == '__main__' ):
        timestamp_test ( )


elif(D==9):
    print('WELCOME TO THE METHOD OF FINITE DIFFERENCE USING THE NINE POINT STENCIL')
    #********************************************************FUNCTION 1******************************************************************
    #************************************************************************************************************************************
    #The function InputDomain request the user to input the cordinates of his/her desired polygonal domain anti-clockwise and plots the corresponding geometry
    print('')
    print('')
    def InputDomain():
        A = n = int(input('WHAT IS THE NUMBER OF SIDES YOU WISH TO CONSIDER TO MAKE UP YOUR POLYGONAL DOMAIN ?' + ' '))
        print('PLEASE CHOOSE HOW YOU WANT TO COLLECT THE INPUT INFORMATION')
        qa = int(input(
            '\033[1;34mPRESS \033[1;34m\n\033[1;38m1:TO COLLECT THE INPUT MANUALLY\033[1;38m\n\n2:TO COLLECT THE INPUT FROM A FILE\n'))
        if qa == 1:
            x = [];
            y = [];
            i = 1;
            while (i < n + 1):
                x1 = float(input('x' + str([i]) + '='));
                y1 = float(input('y' + str([i]) + '='))
                x.append(x1);
                y.append(y1);
                i += 1
            print('x = ', x);
            print('y = ', y);
            f1 = str(input('ENTER THE SOURCE FUNCTION, f(x,y) = '))
            q = input('ENTER THE BOUNDARY FUNCTION, B(x,y)=  ');
            U1 = input('ENTER THE TRUE SOLUTION, u(x,y)=  ')
        else:
            Results = input('Enter the path to the file \n')
            Sheet = open(Results + '.txt', 'r')
            x1 = []
            readFile = open(Results + '.txt', 'r')
            sepFile = readFile.read().split('\n')
            # print(sepFile)
            readFile.close()
            for plotFair in sepFile:
                xa = plotFair.split(',')
                for xb in xa:
                    xc = xb.split(',')
                    x1.append((xc[0]))
            x1.remove('');
            x = [float(x1[0]), float(x1[2]), float(x1[4]), float(x1[6])]
            y = [float(x1[1]), float(x1[3]), float(x1[5]), float(x1[7])];
            f1 = x1[8];
            q = float(x1[9]);
            U1 = x1[10]
            print('x = ',x);
            print('y = ',y)
        plt.xlabel('x-axis', color='red');
        plt.ylabel('y-axis', color='red');
        x.append(x[0]);
        y.append(y[0]);
        plt.xlabel('x-axis', color='red');
        plt.ylabel('y-axis', color='red');
        plt.plot(x, y);
        plt.savefig('Polygonal_Domain.png', transparent=True)
        plt.title('This is a plot of the polygonal domain', color='red')
        plt.show()
        return (x, y, n, f1, q, U1)
    x, y, n, f1, q, U1 = InputDomain()
    print('')

    # ********************************************************FUNCTION 2******************************************************************
    # ************************************************************************************************************************************
    # Function 3 calculates the midpoints of the given geometry
    def Midpoints(x, y, n):
        Mx = [];
        My = []
        for j in range(0, n - 1):
            for k in range(j + 1, n):
                x2 = (x[k] + x[j]) / 2;
                y2 = (y[k] + y[j]) / 2
                Mx.append(x2);
                My.append(y2)
        return (Mx, My)
    Mx, My = Midpoints(x, y, n)
    print('')

    # ----------------------------------------------------FUNCTION 3---------------------------------------
    # ------------------------------------------------------------------------------------------------

    # Function 3 identifies the ccordinates of the calculated midpoints
    def ReorganizingNodes(Mx, My):
        Mxx = [Mx[0], Mx[1], Mx[2], Mx[3], Mx[4], Mx[5]]
        Myy = [My[0], My[1], My[2], My[3], My[4], My[5]]
        #print('Mxx=', Mxx);
        #print('Myy=', Myy)
        return (Mxx, Myy)
    Mxx, Myy = ReorganizingNodes(Mx, My)
    print('')
    # -----------------------------------------------------FUNCTION 4--------------------------------------
    # ------------------------------------------------------------------------------------------------

    # Function 4 performs and plots the first discretization of the given doamin
    def SmallerRecs(x, y, Mxx, Myy):
        CNodes = [];
        CNodesNumb = []
        s1 = [x[0], Mxx[0], Mxx[1], Mxx[2], x[0]], [y[0], Myy[0], Myy[1], Myy[2], y[0]];
        s2 = [Mxx[0], x[1], Mxx[3], Mxx[4], Mxx[0]], [Myy[0], y[1], Myy[3], Myy[4], Myy[0]]
        s3 = [Mxx[2], Mxx[1], Mxx[5], x[3], Mxx[2]], [Myy[2], Myy[1], Myy[5], y[3], Myy[2]];
        s4 = [Mxx[4], Mxx[3], x[2], Mxx[5], Mxx[4]], [Myy[4], Myy[3], y[2], Myy[5], My[4]]
        s = [s1, s2, s3, s4]
        #print('s=', s)
        plt.plot(s1[0], s1[1], s2[0], s2[1], s3[0], s3[1], s4[0], s4[1], color='magenta');
        plt.savefig('First dis.png', transparent=True)
        plt.title('This is the First discretization', color='red')
        plt.show()
        return (s, s1, s2, s3, s4, CNodes, CNodesNumb)
    s, s1, s2, s3, s4, CNodes, CNodesNumb = SmallerRecs(x, y, Mxx, Myy)
    print('')
    # ----------------------------------------------FUNCTION 5-------------------------------------------
    # -----------------------------------------------------------------------------------------------

    # function 5 requests the user to specify the number of partitions he/she will like to consider
    def NumPartition():
        BB = int(input('\033[1;34mPRESS  2\033[1;34m \n  TO DEFINE THE NUMBER OF PARTITIONS OF YOUR DOMAIN:\n'))
        if (BB == 2):
            E = int(input('Enter your desired number of partition along the x-axis:\n n =  '))
            while (E == 0 or E < 0):
                print('\033[1;31m!!!n should be a natural number different from 0!!!\033[1;31m')
                E = int(input('\033[1;38mn=\033[1;38m'))
            F = int(input('Enter yor desired number of partition along the y-axis:\n m =  '))
            while (F == 0 or F < 0):
                print('\033[1;31m!!!m should be a natural number different from 0!!!\033[1;31m')
                F = int(input('\033[1;38m m=\033[1;38m'))
            print("")
        else:
            while (BB != 2):
                print('******ERROR******: PLEASE MAKE SURE YOU ENTER 2.')
                BB = int(input('\033[1;34mPRESS  2\033[1;34m \n  TO DEFINE THE NUMBER OF PARTITIONS OF YOUR DOMAIN:\n'))
                E = int(input('Enter your desired number of partition along the x-axis:\n n =  '))
                while (E == 0 or E < 0):
                    print('\033[1;31m!!!n should be a natural number different from 0!!!\033[1;31m')
                    E = int(input('\033[1;38mn=\033[1;38m'))
                F = int(input('Enter yor desired number of partition along the y-axis:\n m =  '))
                while (F == 0 or F < 0):
                    print('\033[1;31m!!!m should be a natural number different from 0!!!\033[1;31m')
                    F = int(input('\033[1;38mm=\033[1;38m'))
                print("")
        return (E, F, BB)
    E, F, BB = NumPartition()
    print('')

    #----------------------------------------------FUNCTION 6-------------------------------------------
    #-----------------------------------------------------------------------------------------------

    #Function 6 partions the interval of the domain in subintervals of size h and k
    def IntervalPart1(E,F,x,y):
      h = round((x[1] - x[0])/E,8)
      k = round((y[3] - y[0])/F,8)
      print('h=',h); print('k=',k)
      return(h,k)
    h,k = IntervalPart1(E,F,x,y)
    print('')
    #----------------------------------------------FUNCTION 7-------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #Function 7 defines the entire mesh grid including the boundary points
    def TotalgridPoints1(x,y,E,F):
      r = []; s = []
      for j in range (0,F+1):
        for i in range (0,E+1):
          GB = (x[0] + i*h, y[0] + j*k) #definining mesh points
          r.append(round(GB[0],8))
          s.append(round(GB[1],8))
      print('r=', r); print('s =', s)
      return(r,s)
    r,s = TotalgridPoints1(x,y,E,F)
    print('')

    #----------------------------------------------FUNCTION 7-------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #Function 8 initializes a matrix whose entries are zero
    def MatrixInitialize2(E,F):
      L2 = np.zeros([(E+1)*(F+1),(E+1)*(F+1)])
      #B = reshape(A,[(E+1)*(F+1)-1,(E+1)*(F+1)-1])
      return(L2)
    L2 = MatrixInitialize2(E,F)
    print('')

    #----------------------------------------------FUNCTION 8-------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #Function 8 modifies the diagonal and the under diagonal elements
    def ModifyDiagon2(E,F,h,k,L2):
      i = 0;
      while(i < (E+1)*(F+1)):
        p1 = (20/6)*(h*h + k*k)
        p2 = 2*(h*h)*(k*k)
        z = round(sqrt(pow(h,2) + pow(k,2)),8)
        #print('z = ',z)
        L2[i][i]=p1/p2
        i+=1
      return(L2,z)
    L2,z = ModifyDiagon2(E,F,h,k,L2)
    print('')

    #----------------------------------------------FUNCTION 9-------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #Function 9 modifies the under diagonal elements different from zero
    def ModifyUnderDia2(E,F,h,k,r,s,z,L2):
      for j in range (0,(E+1)*(F+1)):
        for i in range(j+1,(E+1)*(F+1)):
          if ((r[i]-r[j]) <= h and s[i]-s[j] == 0):
              L2[i][j] = L2[j][i] = (-4/6)*(pow(h, -2))
          elif ((r[i]-r[j]) == 0 and (s[i]-s[j]) <= k):
              L2[i][j] = L2[j][i] = (-4/6)*(pow(k, -2))
          elif (round(sqrt((r[i]-r[j])**2 + (s[i]-s[j])**2),8) <= z):
            L2[i][j] = L2[j][i] = (-1/6)*(pow(h, -2))
      return(L2)
    L2 = ModifyUnderDia2(E,F,h,k,r,s,z,L2)
    print('')

    #----------------------------------------------FUNCTION 10-------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #Function 10 resuest the user for a source term and the boundary condition
    def SourceFunx(E, F, r, s, f1, q, U1):
        CC = int(input(
            '\033[1;34mPRESS  3\033[1;34m \n  TO INPUT YOUR TRUE SOLUTION, THE SOURCE FUNCTION AND THE BOUNDARY FUNCTION:\n'))
        if (CC == 3):
            #print(U1)
            U = U1
            Ts = []
            for j in range(1, E):
                for i in range(1, F):
                    x = r[i + j * (E + 1)]
                    y = s[i + j * (E + 1)]
                    Ts.append(((eval(str(U1)))))
            print("Ts = ", Ts)
            print('')

            f = eval(f1)
            v = []
            for j in range(1, E):
                for i in range(1, F):
                    x = r[i + j * (E + 1)]
                    y = s[i + j * (E + 1)]
                    (eval(str(f1)))
                    v.append(float((eval(str(f1)))))
            print('v = ', v)
            print('')
            u = eval(str(q))
        else:
            # print('ERROR')
            while (CC != 3):
                print('******ERROR******, PLEASE MAKE SURE YOU ENTER 3.')
                CC = int(input(
                    '\033[1;34mPRESS  3\033[1;34m \n  TO INPUT YOUR TRUE SOLUTION, THE SOURCE FUNCTION AND THE BOUNDARY FUNCTION:\n'))
                # if(CC==3):
            Ts = []
            for j in range(1, E):
                for i in range(1, F):
                    x = r[i + j * (E + 1)]
                    y = s[i + j * (E + 1)]
                    (eval(str(U1)))
                    Ts.append(float((eval(str(U1)))))
            print("Ts = ", Ts)

            f = eval(f1)
            v = []
            for j in range(1, E):
                for i in range(1, F):
                    x = r[i + j * (E + 1)]
                    y = s[i + j * (E + 1)]
                    # g=(eval(str(f))) ;print('g=', g); g1 = round(g, 4) ; print('g1=', g1)
                    v.append(float(eval(str(f1))))
            print('v = ', v)
            print('')
            # u = eval(q)
            print('Boundary segment 1 Numbering')
            b1 = []
            for i in range(0, E):
                x = r[i]
                y = s[i]
                # (eval(u))
                b1.append(float((eval(q))))
                # print(b1)

            # print('Boundary segment 2 Numbering')
            b2 = []
            for j in range(0, F + 1):
                i = j + (j + 1) * E
                x = r[i]
                y = s[i]
                # (eval(u))
                b2.append(float((eval(q))))
                # print(b2)

            # print('Boundary segment 3 Numbering')
            b3 = []
            for i in range(0, E):
                j = i + (E + 1) * F
                x = r[j]
                y = s[j]
                # (eval(u))
                b3.append(float((eval(q))))
                # print(b3)

            # print('Boundary segment 4 Numbering')
            b4 = []
            for j in range(1, F):
                i = j + j * E
                x = r[i]
                y = s[i]
                # (eval(u))
                b4.append(float((eval(q))))
                # print(b4)
        return (Ts, f, u, v, x, y, CC, U)
    Ts, f, u, v, x, y, CC, U = SourceFunx(E, F, r, s, f1, q, U1)
    print('')
    #----------------------------------------------FUNCTION 11-------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #Function 11 request the user for a source term and the boundary condition
    def SystemMatrix(L2):
      (A) =  ModifyDiagon2(E,F,h,k,L2)
      #4print('System Matrix = ',A)
      return((A))
    (A) = SystemMatrix(L2)
    print('')

    #----------------------------------------------FUNCTION 12-------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #Function 12 computes the reduced matrix

    def ReducedMatrix2(E,F,L2):
      print("REDUCED MATRIX")
      A = L2
      s1 = (E + 1) * (F + 1)
      for j in range(0, F + 2):
        j = 0
        A = np.delete(A, (j), axis=0)
        A = np.delete(A, (j), axis=1)

      k1 = s1 - 1 - (F + 2)
      for j in range(0, F + 2):
        A = np.delete(A, (k1), axis=0)
        A = np.delete(A, (k1), axis=1)
        k1 = k1 - 1

      for k1 in range(1, E - 1):
        for j in range(0, 2):
          A = np.delete(A, k1 * (E - 1), axis=0)
          A = np.delete(A, k1 * (E - 1), axis=1)
      print(A)
      return(A)
    A = ReducedMatrix2(E,F,L2)
    print('')


    #----------------------------------------------FUNCTION 13-------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #Function 13 computes the correction of the right hand side using the boundary
    def Correction(E,F,r,s,v,u):
      #print('CORRECTING THE RIGHT HAND SIDE USING BOUNDARY SEGMENTS')
      print('')
      #b = []
      print('')
      #print("WITH BOUNDARY SEGMENT 1")
      print()
      b = []
      p1 = 0
      for j in range (1,F):
        for i in range (1,E):
          x = r[i]
          y = s[i]
          q1 = i+j*(E+1)
          q2 = float(eval(str(u)))
          v[p1] = v[p1] - (q2*L2[q1][i])
          b.append(v[p1])
          p1+=1
          #print(b)
      print('')
      #print("WITH BOUNDARY SEGMENT 2")
      b = []
      p2 = 0
      for j in range (1,F):
        for i in range (1,E):
          x = r[E+j*(E+1)]
          y = s[E+j*(E+1)]
          q3 = i+j*(E+1)
          q4 = E+j*(E+1)
          q5 = float(eval(str(u)))
          v[p2] = v[p2] - (q5*L2[q3][q4])
          b.append(v[p2])
          p2+=1
          #print(b)
      print('')
      #print("WITH BOUNDARY SEGMENT 3")
      b = []
      p3 = 0
      for j in range (1,F):
        for i in range (1,E):
          x = r[j*(E+1)]
          y = s[j*(E+1)]
          q6 = i+j*(E+1)
          q7 = j*(E+1)
          q8 = float(eval(str(u)))
          v[p3] = v[p3] - (q8*L2[q6][q7])
          b.append(v[p3])
          p3+=1
          #print(b)

      print('')
      #print("WITH BOUNDARY SEGMENT 4")
      b = []
      p4 = 0
      for j in range (1,F):
        for i in range (1,E):
          x = r[j*(E+1)]
          y = s[j*(E+1)]
          q9 = i+j*(E+1)
          q10 = j*(E+1)
          q11 = float(eval(str(u)))
          v[p4] = v[p4] - (q11*L2[q9][q10])
          b.append(v[p4])
          p4+=1
          #print(b)
      print('b = ', b)
      return(x,y,b)
    x,y,b = Correction(E,F,r,s,v,u)
    print('')

    #----------------------------------------------FUNCTION 14-------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #Function 13 defines the direct and iterative system solvers
    import numpy as np
    def true(x, y):
        TS1 = (x ** 2 - 1) * (y ** 2 - 1) * exp(x * y)
        return TS1
    TS1 = true(x, y)

    def App_Solution(A, E, F, v, b, Ts, r, s, TS1):

        # exact solution plot

        f = plt.figure()
        a = f.gca(projection='3d')
        x = np.arange(-1, 1, 0.01)
        y = np.arange(-1, 1, 0.01)
        X, Z = np.meshgrid(x, y)
        u = (x**2-1)**2*(y**2-1)**2*sin(x)*cos(y) #(x**2-1)*(y**2-1)*exp(x*y)
        s = a.plot_surface(X,Z,u, cmap=cm.jet)
        f.colorbar(s, shrink=0.5);
        plt.savefig('True_solution22.png', transparent=True)
        plt.title('This is a plot of the true solution', color='red')
        plt.xlabel('x-axis', color='red');
        plt.ylabel('y-axis', color='red')
        plt.show()

        print(
            '\033[1;35m!!!! WELCOME TO OUR PROGRAM FOR SOLVING SYSTEMS OF EQUATIONS (Ax=b) USING EITHER A DIRECT METHOD OR AN ITERATIVE METHOD !!!!\033[1;35m\n\n')
        DD = int(input('\033[1;34mPRESS\033[1;34m \n 4: TO ACCESS AND CHOOSE YOUR DESIRED SOLVER:\n'))
        D = int(input(
            '\033[1;34mPRESS\033[1;34m \n\033[1;38m  1:FOR A DIRECT METHOD \n\n 2: FOR AN ITERATIVE METHOD \n \033[1;38m'));
        q = D
        if (D == 1):
            D = int(input(
                '\033[1;34mPRESS\033[1;34m \n \033[1;38m 1:FOR QR DECOMPOSITION\033[1;38m \033[1;31m)\033[1;31m\n\n \033[1;38m 2:FOR LU FACTORIZATION\033[1;38m\033[1;31m\n\n \033[1;38m 3:FOR CHOLESKY FACTORIZATION\033[1;38m \n'))
            q1 = D
            if (D == 1):
                print('!!\033[1;33mYOUR CHOICE IS QR DECOMPOSITION!!\033[1;33m')
            elif (D == 2):
                print('!!\033[1;33mYOUR CHOICE IS LU DECOMPOSITION!!\033[1;33m')
            elif (D == 3):
                print(
                    '!!\033[1;33mYOUR CHOICE IS CHOLESKY DECOMPOSITION!!(Small tip:This decomposition can be used as a test for positive definity and symmetry of a matrix)!!\033[1;33m')
            else:
                print(
                    '\033[1;31m!!!!!!!!!ERROR MESSAGE!!!!ENTER EITHER 1,2 OR 3 TO MAKE A CHOICE FOR YOUR DIRECT METHOD!!!!!!!!!\033[1;31m')
                exit()
        elif (D == 2):
            D = int(input(
                '\033[1;34mPRESS\033[1;34m \n \033[1;38m 1:FOR JACOBI ITERATION METHOD \n\n 2:FOR GAUSS-SEIDEL ITERATION METHOD \n\n 3:FOR CONJUGATE GRADIENT METHOD \033[1;38m \n'))
            q1 = D

            if (D == 1):
                print('!!\033[1;33mYOUR CHOICE IS JACOBI ITERATION METHOD\033[1;33m!!')
            elif (D == 2):
                print('!!\033[1;33mYOUR CHOICE IS GAUSS-SEIDEL ITERATION METHOD\033[1;33m!!')
            elif (D == 3):
                print('!!\033[1;33mYOUR CHOICE IS CONJUGATE GRADIENT METHOD\033[1;33m!!')
            else:
                print(
                    '\033[1;31m!!!!!!!!ERROR MESSAGE!!!!YOU SHOULD ENTER EITHER 1 OR 2 TO MAKE A CHOICE\033[1;31m!!!!!!!!')
                exit()
        else:
            print('\033[1;31m!!!!!!!ERROR MESSAGE!!!!!ENTER EITHER 1 OR 2 TO CHOOSE A METHOD.!!!!!\033[1;31m')
            exit()
        print('\n')

        # SOLVING SYSTEM QR DECOMPOSITION
        print("\033[1;34mQR DECOMPOSITION\033[1;m")
        Q, R = scipy.linalg.qr(A)
        '''
        print("A:")
        pprint.pprint(A)

        print("Q:\n")
        pprint.pprint(Q)

        print("R:\n")
        pprint.pprint(R)

        A = dot(Q,R) #verifying that A = QR
        print("A:")
        pprint.pprint(A)

        print("b:")
        pprint.pprint(b)
        '''
        print()

        def QRDecomposition(A, b):
            Q, R = scipy.linalg.qr(A)
            y = scipy.linalg.solve(Q, b)
            # print(y)
            xs = scipy.linalg.solve(R, y)
            print('\033[1;48mQRDecomposition result is xs = \n %s\033[1;m' % xs)
            i = 0;
            j = 0;
            zzss = []
            while (i < len(xs) and j < len(Ts)):
                zzs = (Ts[j] - xs[i]);
                zzss.append(zzs)
                i += 1;
                j += 1
            # print('zzss = ', zzss)
            error1 = np.linalg.norm((zzss), ord=inf, axis=None, keepdims=False)
            print("\033[1;32m Error1= \033[1;m:", error1)
            error2 = maximum.reduce(fabs(zzss))
            print("\033[1;32m Error2= \033[1;m:", error2)
            return xs, error1, error2

        # x = QRDecomposition(A,b)
        print("***************************************************************************************")
        print("***************************************************************************************")

        # SOLVING SYSTEM USING LU DECOMPOSITION
        print("\033[1;34mLU DECOMPOSITION\033[1;m")

        def LUFactorization(A, b):
            P, L, U = scipy.linalg.lu(A)
            y = np.linalg.solve(L, b);
            xs = np.linalg.solve(U, y)  # Forward substitution and backward elimination
            print('\033[1;48mLUFactorization result is xs = \n %s\033[1;m' % xs)
            i = 0;
            j = 0;
            zzss = []
            while (i < len(xs) and j < len(Ts)):
                zzs = (Ts[j] - xs[i]);
                zzss.append(zzs)
                i += 1;
                j += 1
            # print('zzss = ', zzss)
            error1 = np.linalg.norm((zzss), ord=inf, axis=None, keepdims=False)
            print("\033[1;32m Error1= \033[1;m:", error1)
            error2 = maximum.reduce(fabs(zzss))
            print("\033[1;32m Error2= \033[1;m:", error2)
            return xs, error1, error2

        # x = LUFactorization(A, b)
        print("***************************************************************************************")
        print("***************************************************************************************")

        # SOLVING THE SYSTEM USING CHOLESKY DECOMPOSITION
        print("\033[1;34mCHOLESKY DECOMPOSITION\033[1;m")

        def CholeskyDecomposition(A, b):
            L = scipy.linalg.cholesky(A, lower=True);
            LT = scipy.linalg.cholesky(A, lower=False)
            # print(L);print(LT);
            y = np.linalg.solve(L, b);
            xs = np.linalg.solve(LT, y)
            print('\033[1;48mCholeskyDecomposition result is xs = \n %s\033[1;m' % xs)
            i = 0;
            j = 0;
            zzss = []
            while (i < len(xs) and j < len(Ts)):
                zzs = (Ts[j] - xs[i]);
                zzss.append(zzs)
                i += 1;
                j += 1
            # print('zzss = ', zzss)
            error1 = np.linalg.norm((zzss), ord=inf, axis=None, keepdims=False)
            print("\033[1;32m Error1= \033[1;m:", error1)
            error2 = maximum.reduce(fabs(zzss))
            print("\033[1;32m Error2= \033[1;m:", error2)
            return xs, error1, error2

        # x = CholeskyDecomposition(A,b)
        print("***************************************************************************************")
        print("***************************************************************************************")

        ITERATION_LIMIT = 1000

        # ITERATIVE METHODS.
        # SOLVING SYSTEM USING JACOBI ITERATION
        def JacobiIteration(A, b):
            print('\033[1;34m  JACOBI ITERATION\033[1;m')
            '''
            print("System:")
            for i in range(A.shape[0]):
                row = ["{}*xs{}".format(A[i, j], j + 1) for j in range(A.shape[1])]
                print(" + ".join(row), "=", b[i])
                '''
            print()

            xs = np.zeros_like(b)
            for it_count in range(ITERATION_LIMIT):
                print("Current solution:", xs)
                xs_new = np.zeros_like(xs)

                for i in range(A.shape[0]):
                    s1 = np.dot(A[i, :i], xs[:i])
                    s2 = np.dot(A[i, i + 1:], xs[i + 1:])
                    xs_new[i] = (b[i] - s1 - s2) / A[i, i]

                if np.allclose(xs, xs_new, atol=1e-10):
                    break

                xs = xs_new

            print("Solution, xs =", xs)
            print("")
            i = 0;
            j = 0;
            zzss = []
            while (i < len(xs) and j < len(Ts)):
                zzs = (Ts[j] - xs[i]);
                zzss.append(zzs)
                i += 1;
                j += 1
            # print('zzss = ', zzss)
            error1 = np.linalg.norm((zzss), ord=inf, axis=None, keepdims=False)
            print("\033[1;32m Error1= \033[1;m:", error1)
            error2 = maximum.reduce(fabs(zzss))
            print("\033[1;32m Error2= \033[1;m:", error2)
            rate = log2(0.0023097703688/ 0.000576444485371);
            print('rate = ', rate)
            return xs, error1, error2

        # x,error = JacobiIteration(A,b)

        print("***************************************************************************************")
        print("***************************************************************************************")
        print('\n\n')

        # SOLVING SYSTEM USING GAUSS-SIEDEL ITERATION
        def GaussSiedelIteration(A, b):
            print('\033[1;34m  GAUSS-SEIDEL ITERATION\033[1;m')

            ITERATION_LIMIT = 1000
            # prints the system
            '''
            print("System:")
            for i in range(A.shape[0]):
                row = ["{}*xs{}".format(A[i, j], j + 1) for j in range(A.shape[1])]
                print(" + ".join(row), "=", b[i])
                '''
            print()

            xs = np.zeros_like(b)
            for it_count in range(ITERATION_LIMIT):
                print("Current solution:", xs)
                xs_new = np.zeros_like(xs)

                for i in range(A.shape[0]):
                    s1 = np.dot(A[i, :i], xs_new[:i])
                    s2 = np.dot(A[i, i + 1:], xs[i + 1:])
                    xs_new[i] = (b[i] - s1 - s2) / A[i, i]

                if np.allclose(xs, xs_new, rtol=1e-8):
                    break

                xs = xs_new

            print("Solution,xs = ", xs)
            print("")
            i = 0;
            j = 0;
            zzss = []
            while (i < len(xs) and j < len(Ts)):
                zzs = (Ts[j] - xs[i]);
                zzss.append(zzs)
                i += 1;
                j += 1
            # print('zzss = ', zzss)
            error1 = np.linalg.norm((zzss), ord=inf, axis=None, keepdims=False)
            print("\033[1;32m Error1= \033[1;m:", error1)
            error2 = maximum.reduce(fabs(zzss))
            print("\033[1;32m Error2= \033[1;m:", error2)
            rate = log2(0.0023097703688/ 0.000576444485371);
            print('rate = ', rate)
            return xs, error1, error2

        # x,error = GaussSiedelIteration(A,b)
        print("***************************************************************************************")
        print("***************************************************************************************")
        print('\n\n')

        def conjugategradient(A, b):
            sheet=open('Approximate solution.txt','w' )
            print('CONJUGATE GRADIENT METHOD')
            xs = scipy.sparse.linalg.cg(A, b, tol=1e-20)
            print("Solution, xs = ", xs)
            # xs1 = array(xs[0])
            # xs2 = reshape(xs1,[(E-1)*(F-1),1])
            # print("Solution, xs2 = ", xs2)
            print("")
            # error = np.dot(A, x) - b
            # error = maximum.reduce(fabs(Ts - xs[0]))
            i = 0;
            j = 0;
            zzss = []
            while (i < len(xs[0]) and j < len(Ts)):
                zzs = (Ts[j] - xs[0][i]);
                zzss.append(zzs)
                i += 1;
                j += 1
            # print('zzss = ', zzss)
            error1 = np.linalg.norm((zzss), ord=inf, axis=None, keepdims=False)
            print("\033[1;32m Error1= \033[1;m:", error1)
            error2 = maximum.reduce(fabs(zzss))
            print("\033[1;32m Error2= \033[1;m:", error2)
            rate = log2(0.0023097703688/ 0.000576444485371);
            print('rate = ', rate)
            return xs, error1, error2

        print("***************************************************************************************")
        print("***************************************************************************************")
        print('\n\n')
        if ((q, q1) == (1, 1)):
            xs, error1, error2 = QRDecomposition(A, b)
        elif ((q, q1) == (1, 2)):
            xs, error1, error2 = LUFactorization(A, b)
        elif ((q, q1) == (1, 3)):
            xs, error1, error2 = CholeskyDecomposition(A, b)
        elif ((q, q1) == (2, 1)):
            xs, error1, error2 = JacobiIteration(A, b)
        elif ((q, q1) == (2, 2)):
            xs, error1, error2 = GaussSiedelIteration(A, b)
        else:
            xs, error1, error2 = conjugategradient(A, b)
        return (D, DD, xs, error1, error2)
    D, DD, xs, error1, error2 = App_Solution(A, E, F, v, b, Ts, r, s, TS1)
    print('')
    print('')
    
    
    f = plt.figure()
    a = f.gca(projection='3d')
    from scipy.interpolate import griddata
    grid_x , grid_y = np.mgrid[-1:1:10j, -1:1:10j]
    points = [[-0.9375, -0.9375], [-0.875, -0.9375], [-0.8125, -0.9375], [-0.75, -0.9375], [-0.6875, -0.9375], [-0.625, -0.9375], [-0.5625, -0.9375], [-0.5, -0.9375], [-0.4375, -0.9375], [-0.375, -0.9375], [-0.3125, -0.9375], [-0.25, -0.9375], [-0.1875, -0.9375], [-0.125, -0.9375], [-0.0625, -0.9375], [0.0, -0.9375], [0.0625, -0.9375], [0.125, -0.9375], [0.1875, -0.9375], [0.25, -0.9375], [0.3125, -0.9375], [0.375, -0.9375], [0.4375, -0.9375], [0.5, -0.9375], [0.5625, -0.9375], [0.625, -0.9375], [0.6875, -0.9375], [0.75, -0.9375], [0.8125, -0.9375], [0.875, -0.9375], [0.9375, -0.9375], [-0.9375, -0.875], [-0.875, -0.875], [-0.8125, -0.875], [-0.75, -0.875], [-0.6875, -0.875], [-0.625, -0.875], [-0.5625, -0.875], [-0.5, -0.875], [-0.4375, -0.875], [-0.375, -0.875], [-0.3125, -0.875], [-0.25, -0.875], [-0.1875, -0.875], [-0.125, -0.875], [-0.0625, -0.875], [0.0, -0.875], [0.0625, -0.875], [0.125, -0.875], [0.1875, -0.875], [0.25, -0.875], [0.3125, -0.875], [0.375, -0.875], [0.4375, -0.875], [0.5, -0.875], [0.5625, -0.875], [0.625, -0.875], [0.6875, -0.875], [0.75, -0.875], [0.8125, -0.875], [0.875, -0.875], [0.9375, -0.875], [-0.9375, -0.8125], [-0.875, -0.8125], [-0.8125, -0.8125], [-0.75, -0.8125], [-0.6875, -0.8125], [-0.625, -0.8125], [-0.5625, -0.8125], [-0.5, -0.8125], [-0.4375, -0.8125], [-0.375, -0.8125], [-0.3125, -0.8125], [-0.25, -0.8125], [-0.1875, -0.8125], [-0.125, -0.8125], [-0.0625, -0.8125], [0.0, -0.8125], [0.0625, -0.8125], [0.125, -0.8125], [0.1875, -0.8125], [0.25, -0.8125], [0.3125, -0.8125], [0.375, -0.8125], [0.4375, -0.8125], [0.5, -0.8125], [0.5625, -0.8125], [0.625, -0.8125], [0.6875, -0.8125], [0.75, -0.8125], [0.8125, -0.8125], [0.875, -0.8125], [0.9375, -0.8125], [-0.9375, -0.75], [-0.875, -0.75], [-0.8125, -0.75], [-0.75, -0.75], [-0.6875, -0.75], [-0.625, -0.75], [-0.5625, -0.75], [-0.5, -0.75], [-0.4375, -0.75], [-0.375, -0.75], [-0.3125, -0.75], [-0.25, -0.75], [-0.1875, -0.75], [-0.125, -0.75], [-0.0625, -0.75], [0.0, -0.75], [0.0625, -0.75], [0.125, -0.75], [0.1875, -0.75], [0.25, -0.75], [0.3125, -0.75], [0.375, -0.75], [0.4375, -0.75], [0.5, -0.75], [0.5625, -0.75], [0.625, -0.75], [0.6875, -0.75], [0.75, -0.75], [0.8125, -0.75], [0.875, -0.75], [0.9375, -0.75], [-0.9375, -0.6875], [-0.875, -0.6875], [-0.8125, -0.6875], [-0.75, -0.6875], [-0.6875, -0.6875], [-0.625, -0.6875], [-0.5625, -0.6875], [-0.5, -0.6875], [-0.4375, -0.6875], [-0.375, -0.6875], [-0.3125, -0.6875], [-0.25, -0.6875], [-0.1875, -0.6875], [-0.125, -0.6875], [-0.0625, -0.6875], [0.0, -0.6875], [0.0625, -0.6875], [0.125, -0.6875], [0.1875, -0.6875], [0.25, -0.6875], [0.3125, -0.6875], [0.375, -0.6875], [0.4375, -0.6875], [0.5, -0.6875], [0.5625, -0.6875], [0.625, -0.6875], [0.6875, -0.6875], [0.75, -0.6875], [0.8125, -0.6875], [0.875, -0.6875], [0.9375, -0.6875], [-0.9375, -0.625], [-0.875, -0.625], [-0.8125, -0.625], [-0.75, -0.625], [-0.6875, -0.625], [-0.625, -0.625], [-0.5625, -0.625], [-0.5, -0.625], [-0.4375, -0.625], [-0.375, -0.625], [-0.3125, -0.625], [-0.25, -0.625], [-0.1875, -0.625], [-0.125, -0.625], [-0.0625, -0.625], [0.0, -0.625], [0.0625, -0.625], [0.125, -0.625], [0.1875, -0.625], [0.25, -0.625], [0.3125, -0.625], [0.375, -0.625], [0.4375, -0.625], [0.5, -0.625], [0.5625, -0.625], [0.625, -0.625], [0.6875, -0.625], [0.75, -0.625], [0.8125, -0.625], [0.875, -0.625], [0.9375, -0.625], [-0.9375, -0.5625], [-0.875, -0.5625], [-0.8125, -0.5625], [-0.75, -0.5625], [-0.6875, -0.5625], [-0.625, -0.5625], [-0.5625, -0.5625], [-0.5, -0.5625], [-0.4375, -0.5625], [-0.375, -0.5625], [-0.3125, -0.5625], [-0.25, -0.5625], [-0.1875, -0.5625], [-0.125, -0.5625], [-0.0625, -0.5625], [0.0, -0.5625], [0.0625, -0.5625], [0.125, -0.5625], [0.1875, -0.5625], [0.25, -0.5625], [0.3125, -0.5625], [0.375, -0.5625], [0.4375, -0.5625], [0.5, -0.5625], [0.5625, -0.5625], [0.625, -0.5625], [0.6875, -0.5625], [0.75, -0.5625], [0.8125, -0.5625], [0.875, -0.5625], [0.9375, -0.5625], [-0.9375, -0.5], [-0.875, -0.5], [-0.8125, -0.5], [-0.75, -0.5], [-0.6875, -0.5], [-0.625, -0.5], [-0.5625, -0.5], [-0.5, -0.5], [-0.4375, -0.5], [-0.375, -0.5], [-0.3125, -0.5], [-0.25, -0.5], [-0.1875, -0.5], [-0.125, -0.5], [-0.0625, -0.5], [0.0, -0.5], [0.0625, -0.5], [0.125, -0.5], [0.1875, -0.5], [0.25, -0.5], [0.3125, -0.5], [0.375, -0.5], [0.4375, -0.5], [0.5, -0.5], [0.5625, -0.5], [0.625, -0.5], [0.6875, -0.5], [0.75, -0.5], [0.8125, -0.5], [0.875, -0.5], [0.9375, -0.5], [-0.9375, -0.4375], [-0.875, -0.4375], [-0.8125, -0.4375], [-0.75, -0.4375], [-0.6875, -0.4375], [-0.625, -0.4375], [-0.5625, -0.4375], [-0.5, -0.4375], [-0.4375, -0.4375], [-0.375, -0.4375], [-0.3125, -0.4375], [-0.25, -0.4375], [-0.1875, -0.4375], [-0.125, -0.4375], [-0.0625, -0.4375], [0.0, -0.4375], [0.0625, -0.4375], [0.125, -0.4375], [0.1875, -0.4375], [0.25, -0.4375], [0.3125, -0.4375], [0.375, -0.4375], [0.4375, -0.4375], [0.5, -0.4375], [0.5625, -0.4375], [0.625, -0.4375], [0.6875, -0.4375], [0.75, -0.4375], [0.8125, -0.4375], [0.875, -0.4375], [0.9375, -0.4375], [-0.9375, -0.375], [-0.875, -0.375], [-0.8125, -0.375], [-0.75, -0.375], [-0.6875, -0.375], [-0.625, -0.375], [-0.5625, -0.375], [-0.5, -0.375], [-0.4375, -0.375], [-0.375, -0.375], [-0.3125, -0.375], [-0.25, -0.375], [-0.1875, -0.375], [-0.125, -0.375], [-0.0625, -0.375], [0.0, -0.375], [0.0625, -0.375], [0.125, -0.375], [0.1875, -0.375], [0.25, -0.375], [0.3125, -0.375], [0.375, -0.375], [0.4375, -0.375], [0.5, -0.375], [0.5625, -0.375], [0.625, -0.375], [0.6875, -0.375], [0.75, -0.375], [0.8125, -0.375], [0.875, -0.375], [0.9375, -0.375], [-0.9375, -0.3125], [-0.875, -0.3125], [-0.8125, -0.3125], [-0.75, -0.3125], [-0.6875, -0.3125], [-0.625, -0.3125], [-0.5625, -0.3125], [-0.5, -0.3125], [-0.4375, -0.3125], [-0.375, -0.3125], [-0.3125, -0.3125], [-0.25, -0.3125], [-0.1875, -0.3125], [-0.125, -0.3125], [-0.0625, -0.3125], [0.0, -0.3125], [0.0625, -0.3125], [0.125, -0.3125], [0.1875, -0.3125], [0.25, -0.3125], [0.3125, -0.3125], [0.375, -0.3125], [0.4375, -0.3125], [0.5, -0.3125], [0.5625, -0.3125], [0.625, -0.3125], [0.6875, -0.3125], [0.75, -0.3125], [0.8125, -0.3125], [0.875, -0.3125], [0.9375, -0.3125], [-0.9375, -0.25], [-0.875, -0.25], [-0.8125, -0.25], [-0.75, -0.25], [-0.6875, -0.25], [-0.625, -0.25], [-0.5625, -0.25], [-0.5, -0.25], [-0.4375, -0.25], [-0.375, -0.25], [-0.3125, -0.25], [-0.25, -0.25], [-0.1875, -0.25], [-0.125, -0.25], [-0.0625, -0.25], [0.0, -0.25], [0.0625, -0.25], [0.125, -0.25], [0.1875, -0.25], [0.25, -0.25], [0.3125, -0.25], [0.375, -0.25], [0.4375, -0.25], [0.5, -0.25], [0.5625, -0.25], [0.625, -0.25], [0.6875, -0.25], [0.75, -0.25], [0.8125, -0.25], [0.875, -0.25], [0.9375, -0.25], [-0.9375, -0.1875], [-0.875, -0.1875], [-0.8125, -0.1875], [-0.75, -0.1875], [-0.6875, -0.1875], [-0.625, -0.1875], [-0.5625, -0.1875], [-0.5, -0.1875], [-0.4375, -0.1875], [-0.375, -0.1875], [-0.3125, -0.1875], [-0.25, -0.1875], [-0.1875, -0.1875], [-0.125, -0.1875], [-0.0625, -0.1875], [0.0, -0.1875], [0.0625, -0.1875], [0.125, -0.1875], [0.1875, -0.1875], [0.25, -0.1875], [0.3125, -0.1875], [0.375, -0.1875], [0.4375, -0.1875], [0.5, -0.1875], [0.5625, -0.1875], [0.625, -0.1875], [0.6875, -0.1875], [0.75, -0.1875], [0.8125, -0.1875], [0.875, -0.1875], [0.9375, -0.1875], [-0.9375, -0.125], [-0.875, -0.125], [-0.8125, -0.125], [-0.75, -0.125], [-0.6875, -0.125], [-0.625, -0.125], [-0.5625, -0.125], [-0.5, -0.125], [-0.4375, -0.125], [-0.375, -0.125], [-0.3125, -0.125], [-0.25, -0.125], [-0.1875, -0.125], [-0.125, -0.125], [-0.0625, -0.125], [0.0, -0.125], [0.0625, -0.125], [0.125, -0.125], [0.1875, -0.125], [0.25, -0.125], [0.3125, -0.125], [0.375, -0.125], [0.4375, -0.125], [0.5, -0.125], [0.5625, -0.125], [0.625, -0.125], [0.6875, -0.125], [0.75, -0.125], [0.8125, -0.125], [0.875, -0.125], [0.9375, -0.125], [-0.9375, -0.0625], [-0.875, -0.0625], [-0.8125, -0.0625], [-0.75, -0.0625], [-0.6875, -0.0625], [-0.625, -0.0625], [-0.5625, -0.0625], [-0.5, -0.0625], [-0.4375, -0.0625], [-0.375, -0.0625], [-0.3125, -0.0625], [-0.25, -0.0625], [-0.1875, -0.0625], [-0.125, -0.0625], [-0.0625, -0.0625], [0.0, -0.0625], [0.0625, -0.0625], [0.125, -0.0625], [0.1875, -0.0625], [0.25, -0.0625], [0.3125, -0.0625], [0.375, -0.0625], [0.4375, -0.0625], [0.5, -0.0625], [0.5625, -0.0625], [0.625, -0.0625], [0.6875, -0.0625], [0.75, -0.0625], [0.8125, -0.0625], [0.875, -0.0625], [0.9375, -0.0625], [-0.9375, 0.0], [-0.875, 0.0], [-0.8125, 0.0], [-0.75, 0.0], [-0.6875, 0.0], [-0.625, 0.0], [-0.5625, 0.0], [-0.5, 0.0], [-0.4375, 0.0], [-0.375, 0.0], [-0.3125, 0.0], [-0.25, 0.0], [-0.1875, 0.0], [-0.125, 0.0], [-0.0625, 0.0], [0.0, 0.0], [0.0625, 0.0], [0.125, 0.0], [0.1875, 0.0], [0.25, 0.0], [0.3125, 0.0], [0.375, 0.0], [0.4375, 0.0], [0.5, 0.0], [0.5625, 0.0], [0.625, 0.0], [0.6875, 0.0], [0.75, 0.0], [0.8125, 0.0], [0.875, 0.0], [0.9375, 0.0], [-0.9375, 0.0625], [-0.875, 0.0625], [-0.8125, 0.0625], [-0.75, 0.0625], [-0.6875, 0.0625], [-0.625, 0.0625], [-0.5625, 0.0625], [-0.5, 0.0625], [-0.4375, 0.0625], [-0.375, 0.0625], [-0.3125, 0.0625], [-0.25, 0.0625], [-0.1875, 0.0625], [-0.125, 0.0625], [-0.0625, 0.0625], [0.0, 0.0625], [0.0625, 0.0625], [0.125, 0.0625], [0.1875, 0.0625], [0.25, 0.0625], [0.3125, 0.0625], [0.375, 0.0625], [0.4375, 0.0625], [0.5, 0.0625], [0.5625, 0.0625], [0.625, 0.0625], [0.6875, 0.0625], [0.75, 0.0625], [0.8125, 0.0625], [0.875, 0.0625], [0.9375, 0.0625], [-0.9375, 0.125], [-0.875, 0.125], [-0.8125, 0.125], [-0.75, 0.125], [-0.6875, 0.125], [-0.625, 0.125], [-0.5625, 0.125], [-0.5, 0.125], [-0.4375, 0.125], [-0.375, 0.125], [-0.3125, 0.125], [-0.25, 0.125], [-0.1875, 0.125], [-0.125, 0.125], [-0.0625, 0.125], [0.0, 0.125], [0.0625, 0.125], [0.125, 0.125], [0.1875, 0.125], [0.25, 0.125], [0.3125, 0.125], [0.375, 0.125], [0.4375, 0.125], [0.5, 0.125], [0.5625, 0.125], [0.625, 0.125], [0.6875, 0.125], [0.75, 0.125], [0.8125, 0.125], [0.875, 0.125], [0.9375, 0.125], [-0.9375, 0.1875], [-0.875, 0.1875], [-0.8125, 0.1875], [-0.75, 0.1875], [-0.6875, 0.1875], [-0.625, 0.1875], [-0.5625, 0.1875], [-0.5, 0.1875], [-0.4375, 0.1875], [-0.375, 0.1875], [-0.3125, 0.1875], [-0.25, 0.1875], [-0.1875, 0.1875], [-0.125, 0.1875], [-0.0625, 0.1875], [0.0, 0.1875], [0.0625, 0.1875], [0.125, 0.1875], [0.1875, 0.1875], [0.25, 0.1875], [0.3125, 0.1875], [0.375, 0.1875], [0.4375, 0.1875], [0.5, 0.1875], [0.5625, 0.1875], [0.625, 0.1875], [0.6875, 0.1875], [0.75, 0.1875], [0.8125, 0.1875], [0.875, 0.1875], [0.9375, 0.1875], [-0.9375, 0.25], [-0.875, 0.25], [-0.8125, 0.25], [-0.75, 0.25], [-0.6875, 0.25], [-0.625, 0.25], [-0.5625, 0.25], [-0.5, 0.25], [-0.4375, 0.25], [-0.375, 0.25], [-0.3125, 0.25], [-0.25, 0.25], [-0.1875, 0.25], [-0.125, 0.25], [-0.0625, 0.25], [0.0, 0.25], [0.0625, 0.25], [0.125, 0.25], [0.1875, 0.25], [0.25, 0.25], [0.3125, 0.25], [0.375, 0.25], [0.4375, 0.25], [0.5, 0.25], [0.5625, 0.25], [0.625, 0.25], [0.6875, 0.25], [0.75, 0.25], [0.8125, 0.25], [0.875, 0.25], [0.9375, 0.25], [-0.9375, 0.3125], [-0.875, 0.3125], [-0.8125, 0.3125], [-0.75, 0.3125], [-0.6875, 0.3125], [-0.625, 0.3125], [-0.5625, 0.3125], [-0.5, 0.3125], [-0.4375, 0.3125], [-0.375, 0.3125], [-0.3125, 0.3125], [-0.25, 0.3125], [-0.1875, 0.3125], [-0.125, 0.3125], [-0.0625, 0.3125], [0.0, 0.3125], [0.0625, 0.3125], [0.125, 0.3125], [0.1875, 0.3125], [0.25, 0.3125], [0.3125, 0.3125], [0.375, 0.3125], [0.4375, 0.3125], [0.5, 0.3125], [0.5625, 0.3125], [0.625, 0.3125], [0.6875, 0.3125], [0.75, 0.3125], [0.8125, 0.3125], [0.875, 0.3125], [0.9375, 0.3125], [-0.9375, 0.375], [-0.875, 0.375], [-0.8125, 0.375], [-0.75, 0.375], [-0.6875, 0.375], [-0.625, 0.375], [-0.5625, 0.375], [-0.5, 0.375], [-0.4375, 0.375], [-0.375, 0.375], [-0.3125, 0.375], [-0.25, 0.375], [-0.1875, 0.375], [-0.125, 0.375], [-0.0625, 0.375], [0.0, 0.375], [0.0625, 0.375], [0.125, 0.375], [0.1875, 0.375], [0.25, 0.375], [0.3125, 0.375], [0.375, 0.375], [0.4375, 0.375], [0.5, 0.375], [0.5625, 0.375], [0.625, 0.375], [0.6875, 0.375], [0.75, 0.375], [0.8125, 0.375], [0.875, 0.375], [0.9375, 0.375], [-0.9375, 0.4375], [-0.875, 0.4375], [-0.8125, 0.4375], [-0.75, 0.4375], [-0.6875, 0.4375], [-0.625, 0.4375], [-0.5625, 0.4375], [-0.5, 0.4375], [-0.4375, 0.4375], [-0.375, 0.4375], [-0.3125, 0.4375], [-0.25, 0.4375], [-0.1875, 0.4375], [-0.125, 0.4375], [-0.0625, 0.4375], [0.0, 0.4375], [0.0625, 0.4375], [0.125, 0.4375], [0.1875, 0.4375], [0.25, 0.4375], [0.3125, 0.4375], [0.375, 0.4375], [0.4375, 0.4375], [0.5, 0.4375], [0.5625, 0.4375], [0.625, 0.4375], [0.6875, 0.4375], [0.75, 0.4375], [0.8125, 0.4375], [0.875, 0.4375], [0.9375, 0.4375], [-0.9375, 0.5], [-0.875, 0.5], [-0.8125, 0.5], [-0.75, 0.5], [-0.6875, 0.5], [-0.625, 0.5], [-0.5625, 0.5], [-0.5, 0.5], [-0.4375, 0.5], [-0.375, 0.5], [-0.3125, 0.5], [-0.25, 0.5], [-0.1875, 0.5], [-0.125, 0.5], [-0.0625, 0.5], [0.0, 0.5], [0.0625, 0.5], [0.125, 0.5], [0.1875, 0.5], [0.25, 0.5], [0.3125, 0.5], [0.375, 0.5], [0.4375, 0.5], [0.5, 0.5], [0.5625, 0.5], [0.625, 0.5], [0.6875, 0.5], [0.75, 0.5], [0.8125, 0.5], [0.875, 0.5], [0.9375, 0.5], [-0.9375, 0.5625], [-0.875, 0.5625], [-0.8125, 0.5625], [-0.75, 0.5625], [-0.6875, 0.5625], [-0.625, 0.5625], [-0.5625, 0.5625], [-0.5, 0.5625], [-0.4375, 0.5625], [-0.375, 0.5625], [-0.3125, 0.5625], [-0.25, 0.5625], [-0.1875, 0.5625], [-0.125, 0.5625], [-0.0625, 0.5625], [0.0, 0.5625], [0.0625, 0.5625], [0.125, 0.5625], [0.1875, 0.5625], [0.25, 0.5625], [0.3125, 0.5625], [0.375, 0.5625], [0.4375, 0.5625], [0.5, 0.5625], [0.5625, 0.5625], [0.625, 0.5625], [0.6875, 0.5625], [0.75, 0.5625], [0.8125, 0.5625], [0.875, 0.5625], [0.9375, 0.5625], [-0.9375, 0.625], [-0.875, 0.625], [-0.8125, 0.625], [-0.75, 0.625], [-0.6875, 0.625], [-0.625, 0.625], [-0.5625, 0.625], [-0.5, 0.625], [-0.4375, 0.625], [-0.375, 0.625], [-0.3125, 0.625], [-0.25, 0.625], [-0.1875, 0.625], [-0.125, 0.625], [-0.0625, 0.625], [0.0, 0.625], [0.0625, 0.625], [0.125, 0.625], [0.1875, 0.625], [0.25, 0.625], [0.3125, 0.625], [0.375, 0.625], [0.4375, 0.625], [0.5, 0.625], [0.5625, 0.625], [0.625, 0.625], [0.6875, 0.625], [0.75, 0.625], [0.8125, 0.625], [0.875, 0.625], [0.9375, 0.625], [-0.9375, 0.6875], [-0.875, 0.6875], [-0.8125, 0.6875], [-0.75, 0.6875], [-0.6875, 0.6875], [-0.625, 0.6875], [-0.5625, 0.6875], [-0.5, 0.6875], [-0.4375, 0.6875], [-0.375, 0.6875], [-0.3125, 0.6875], [-0.25, 0.6875], [-0.1875, 0.6875], [-0.125, 0.6875], [-0.0625, 0.6875], [0.0, 0.6875], [0.0625, 0.6875], [0.125, 0.6875], [0.1875, 0.6875], [0.25, 0.6875], [0.3125, 0.6875], [0.375, 0.6875], [0.4375, 0.6875], [0.5, 0.6875], [0.5625, 0.6875], [0.625, 0.6875], [0.6875, 0.6875], [0.75, 0.6875], [0.8125, 0.6875], [0.875, 0.6875], [0.9375, 0.6875], [-0.9375, 0.75], [-0.875, 0.75], [-0.8125, 0.75], [-0.75, 0.75], [-0.6875, 0.75], [-0.625, 0.75], [-0.5625, 0.75], [-0.5, 0.75], [-0.4375, 0.75], [-0.375, 0.75], [-0.3125, 0.75], [-0.25, 0.75], [-0.1875, 0.75], [-0.125, 0.75], [-0.0625, 0.75], [0.0, 0.75], [0.0625, 0.75], [0.125, 0.75], [0.1875, 0.75], [0.25, 0.75], [0.3125, 0.75], [0.375, 0.75], [0.4375, 0.75], [0.5, 0.75], [0.5625, 0.75], [0.625, 0.75], [0.6875, 0.75], [0.75, 0.75], [0.8125, 0.75], [0.875, 0.75], [0.9375, 0.75], [-0.9375, 0.8125], [-0.875, 0.8125], [-0.8125, 0.8125], [-0.75, 0.8125], [-0.6875, 0.8125], [-0.625, 0.8125], [-0.5625, 0.8125], [-0.5, 0.8125], [-0.4375, 0.8125], [-0.375, 0.8125], [-0.3125, 0.8125], [-0.25, 0.8125], [-0.1875, 0.8125], [-0.125, 0.8125], [-0.0625, 0.8125], [0.0, 0.8125], [0.0625, 0.8125], [0.125, 0.8125], [0.1875, 0.8125], [0.25, 0.8125], [0.3125, 0.8125], [0.375, 0.8125], [0.4375, 0.8125], [0.5, 0.8125], [0.5625, 0.8125], [0.625, 0.8125], [0.6875, 0.8125], [0.75, 0.8125], [0.8125, 0.8125], [0.875, 0.8125], [0.9375, 0.8125], [-0.9375, 0.875], [-0.875, 0.875], [-0.8125, 0.875], [-0.75, 0.875], [-0.6875, 0.875], [-0.625, 0.875], [-0.5625, 0.875], [-0.5, 0.875], [-0.4375, 0.875], [-0.375, 0.875], [-0.3125, 0.875], [-0.25, 0.875], [-0.1875, 0.875], [-0.125, 0.875], [-0.0625, 0.875], [0.0, 0.875], [0.0625, 0.875], [0.125, 0.875], [0.1875, 0.875], [0.25, 0.875], [0.3125, 0.875], [0.375, 0.875], [0.4375, 0.875], [0.5, 0.875], [0.5625, 0.875], [0.625, 0.875], [0.6875, 0.875], [0.75, 0.875], [0.8125, 0.875], [0.875, 0.875], [0.9375, 0.875], [-0.9375, 0.9375], [-0.875, 0.9375], [-0.8125, 0.9375], [-0.75, 0.9375], [-0.6875, 0.9375], [-0.625, 0.9375], [-0.5625, 0.9375], [-0.5, 0.9375], [-0.4375, 0.9375], [-0.375, 0.9375], [-0.3125, 0.9375], [-0.25, 0.9375], [-0.1875, 0.9375], [-0.125, 0.9375], [-0.0625, 0.9375], [0.0, 0.9375], [0.0625, 0.9375], [0.125, 0.9375], [0.1875, 0.9375], [0.25, 0.9375], [0.3125, 0.9375], [0.375, 0.9375], [0.4375, 0.9375], [0.5, 0.9375], [0.5625, 0.9375], [0.625, 0.9375], [0.6875, 0.9375], [0.75, 0.9375], [0.8125, 0.9375], [0.875, 0.9375], [0.9375, 0.9375]]



    values = [ -1.54880755e-04,  -4.31245730e-04,  -7.86994400e-04,
        -1.17600488e-03,  -1.55567296e-03,  -1.88955424e-03,
        -2.14851828e-03,  -2.31124185e-03,  -2.36430009e-03,
        -2.30196011e-03,  -2.12573176e-03,  -1.84371315e-03,
        -1.46976147e-03,  -1.02251809e-03,  -5.24316247e-04,
         9.82638092e-18,   5.24316247e-04,   1.02251809e-03,
         1.46976147e-03,   1.84371315e-03,   2.12573176e-03,
         2.30196011e-03,   2.36430009e-03,   2.31124185e-03,
         2.14851828e-03,   1.88955424e-03,   1.55567296e-03,
         1.17600488e-03,   7.86994400e-04,   4.31245730e-04,
         1.54880755e-04,  -5.01188114e-04,  -1.60328937e-03,
        -3.07397267e-03,  -4.70066009e-03,  -6.29669568e-03,
        -7.70562932e-03,  -8.80367166e-03,  -9.50073002e-03,
        -9.74027607e-03,  -9.49820231e-03,  -8.78078463e-03,
        -7.62185563e-03,  -6.07929107e-03,  -4.23091475e-03,
        -2.16993121e-03,   1.97230458e-17,   2.16993121e-03,
         4.23091475e-03,   6.07929107e-03,   7.62185563e-03,
         8.78078463e-03,   9.49820231e-03,   9.74027607e-03,
         9.50073002e-03,   8.80367166e-03,   7.70562932e-03,
         6.29669568e-03,   4.70066009e-03,   3.07397267e-03,
         1.60328937e-03,   5.01188114e-04,  -1.05080590e-03,
        -3.51873044e-03,  -6.85121454e-03,  -1.05519835e-02,
        -1.41898956e-02,  -1.74055950e-02,  -1.99156241e-02,
        -2.15140550e-02,  -2.20718048e-02,  -2.15338189e-02,
        -1.99143093e-02,  -1.72902489e-02,  -1.37933352e-02,
        -9.60064968e-03,  -4.92425762e-03,   2.96246534e-17,
         4.92425762e-03,   9.60064968e-03,   1.37933352e-02,
         1.72902489e-02,   1.99143093e-02,   2.15338189e-02,
         2.20718048e-02,   2.15140550e-02,   1.99156241e-02,
         1.74055950e-02,   1.41898956e-02,   1.05519835e-02,
         6.85121454e-03,   3.51873044e-03,   1.05080590e-03,
        -1.79364751e-03,  -6.12914850e-03,  -1.20150764e-02,
        -1.85643966e-02,  -2.50088067e-02,  -3.07092097e-02,
        -3.51621267e-02,  -3.80020679e-02,  -3.90000180e-02,
        -3.80582661e-02,  -3.52018666e-02,  -3.05670544e-02,
        -2.43869772e-02,  -1.69751381e-02,  -8.70697191e-03,
         3.91405097e-17,   8.70697191e-03,   1.69751381e-02,
         2.43869772e-02,   3.05670544e-02,   3.52018666e-02,
         3.80582661e-02,   3.90000180e-02,   3.80020679e-02,
         3.51621267e-02,   3.07092097e-02,   2.50088067e-02,
         1.85643966e-02,   1.20150764e-02,   6.12914850e-03,
         1.79364751e-03,  -2.70583300e-03,  -9.34471565e-03,
        -1.83836395e-02,  -2.84525304e-02,  -3.83659079e-02,
        -4.71384321e-02,  -5.39942871e-02,  -5.83704845e-02,
        -5.99143054e-02,  -5.84752016e-02,  -5.40915679e-02,
        -4.69728647e-02,  -3.74776378e-02,  -2.60880320e-02,
        -1.33814446e-02,   4.82453366e-17,   1.33814446e-02,
         2.60880320e-02,   3.74776378e-02,   4.69728647e-02,
         5.40915679e-02,   5.84752016e-02,   5.99143054e-02,
         5.83704845e-02,   5.39942871e-02,   4.71384321e-02,
         3.83659079e-02,   2.84525304e-02,   1.83836395e-02,
         9.34471565e-03,   2.70583300e-03,  -3.75365382e-03,
        -1.30435185e-02,  -2.57131462e-02,  -3.98360589e-02,
        -5.37459961e-02,  -6.60584710e-02,  -7.56836724e-02,
        -8.18308821e-02,  -8.40047221e-02,  -8.19936781e-02,
        -7.58514612e-02,  -6.58718732e-02,  -5.25579332e-02,
        -3.65860976e-02,  -1.87664768e-02,   5.76971740e-17,
         1.87664768e-02,   3.65860976e-02,   5.25579332e-02,
         6.58718732e-02,   7.58514612e-02,   8.19936781e-02,
         8.40047221e-02,   8.18308821e-02,   7.56836724e-02,
         6.60584710e-02,   5.37459961e-02,   3.98360589e-02,
         2.57131462e-02,   1.30435185e-02,   3.75365382e-03,
        -4.89646481e-03,  -1.70803150e-02,  -3.37144590e-02,
        -5.22647982e-02,  -7.05399012e-02,  -8.67191973e-02,
        -9.93697772e-02,  -1.07452059e-01,  -1.10314747e-01,
        -1.07679667e-01,  -9.96172159e-02,  -8.65132987e-02,
        -6.90287347e-02,  -4.80522321e-02,  -2.46481070e-02,
         6.78661176e-17,   2.46481070e-02,   4.80522321e-02,
         6.90287347e-02,   8.65132987e-02,   9.96172159e-02,
         1.07679667e-01,   1.10314747e-01,   1.07452059e-01,
         9.93697772e-02,   8.67191973e-02,   7.05399012e-02,
         5.22647982e-02,   3.37144590e-02,   1.70803150e-02,
         4.89646481e-03,  -6.08920088e-03,  -2.12948651e-02,
        -4.20691750e-02,  -6.52434746e-02,  -8.80778059e-02,
        -1.08296066e-01,  -1.24106910e-01,  -1.34210749e-01,
        -1.37793366e-01,  -1.34506912e-01,  -1.24439181e-01,
        -1.08072262e-01,  -8.62317901e-02,  -6.00281748e-02,
        -3.07912618e-02,   7.87996872e-17,   3.07912618e-02,
         6.00281748e-02,   8.62317901e-02,   1.08072262e-01,
         1.24439181e-01,   1.34506912e-01,   1.37793366e-01,
         1.34210749e-01,   1.24106910e-01,   1.08296066e-01,
         8.80778059e-02,   6.52434746e-02,   4.20691750e-02,
         2.12948651e-02,   6.08920088e-03,  -7.28469039e-03,
        -2.55198506e-02,  -5.04451329e-02,  -7.82556691e-02,
        -1.05661503e-01,  -1.29929743e-01,  -1.48909598e-01,
        -1.61040719e-01,  -1.65345487e-01,  -1.61406153e-01,
        -1.49327934e-01,  -1.29689349e-01,  -1.03481294e-01,
        -7.20364856e-02,  -3.69510298e-02,   9.02170885e-17,
         3.69510298e-02,   7.20364856e-02,   1.03481294e-01,
         1.29689349e-01,   1.49327934e-01,   1.61406153e-01,
         1.65345487e-01,   1.61040719e-01,   1.48909598e-01,
         1.29929743e-01,   1.05661503e-01,   7.82556691e-02,
         5.04451329e-02,   2.55198506e-02,   7.28469039e-03,
        -8.43579940e-03,  -2.95883187e-02,  -5.85110665e-02,
        -9.07864988e-02,  -1.22594983e-01,  -1.50763699e-01,
        -1.72795659e-01,  -1.86879355e-01,  -1.91879771e-01,
        -1.87311820e-01,  -1.73297480e-01,  -1.50508132e-01,
        -1.20093829e-01,  -8.36013887e-02,  -4.28833544e-02,
         1.02516294e-16,   4.28833544e-02,   8.36013887e-02,
         1.20093829e-01,   1.50508132e-01,   1.73297480e-01,
         1.87311820e-01,   1.91879771e-01,   1.86879355e-01,
         1.72795659e-01,   1.50763699e-01,   1.22594983e-01,
         9.07864988e-02,   5.85110665e-02,   2.95883187e-02,
         8.43579940e-03,  -9.49739865e-03,  -3.33405541e-02,
        -6.59501700e-02,  -1.02343640e-01,  -1.38212805e-01,
        -1.69979081e-01,  -1.94826161e-01,  -2.10710868e-01,
        -2.16352998e-01,  -2.11205342e-01,  -1.95405327e-01,
        -1.69709977e-01,  -1.35416140e-01,  -9.42681085e-02,
        -4.83549513e-02,   1.13658789e-16,   4.83549513e-02,
         9.42681085e-02,   1.35416140e-01,   1.69709977e-01,
         1.95405327e-01,   2.11205342e-01,   2.16352998e-01,
         2.10710868e-01,   1.94826161e-01,   1.69979081e-01,
         1.38212805e-01,   1.02343640e-01,   6.59501700e-02,
         3.33405541e-02,   9.49739865e-03,  -1.04281322e-02,
        -3.66302776e-02,  -7.24723475e-02,  -1.12476326e-01,
        -1.51905741e-01,  -1.86826246e-01,  -2.14141561e-01,
        -2.31605381e-01,  -2.37810196e-01,  -2.32154324e-01,
        -2.14788729e-01,  -1.86545521e-01,  -1.48850252e-01,
        -1.03620354e-01,  -5.31522783e-02,   1.23846471e-16,
         5.31522783e-02,   1.03620354e-01,   1.48850252e-01,
         1.86545521e-01,   2.14788729e-01,   2.32154324e-01,
         2.37810196e-01,   2.31605381e-01,   2.14141561e-01,
         1.86826246e-01,   1.51905741e-01,   1.12476326e-01,
         7.24723475e-02,   3.66302776e-02,   1.04281322e-02,
        -1.11919633e-02,  -3.93300731e-02,  -7.78249468e-02,
        -1.20792003e-01,  -1.63143262e-01,  -2.00652409e-01,
        -2.29993391e-01,  -2.48753194e-01,  -2.55419826e-01,
        -2.49346891e-01,  -2.30696462e-01,  -2.00362271e-01,
        -1.59875490e-01,  -1.11295651e-01,  -5.70893991e-02,
         1.33538200e-16,   5.70893991e-02,   1.11295651e-01,
         1.59875490e-01,   2.00362271e-01,   2.30696462e-01,
         2.49346891e-01,   2.55419826e-01,   2.48753194e-01,
         2.29993391e-01,   2.00652409e-01,   1.63143262e-01,
         1.20792003e-01,   7.78249468e-02,   3.93300731e-02,
         1.11919633e-02,  -1.17594741e-02,  -4.13359551e-02,
        -8.18017944e-02,  -1.26970347e-01,  -1.71492472e-01,
        -2.10924929e-01,  -2.41770948e-01,  -2.61493651e-01,
        -2.68503416e-01,  -2.62120623e-01,  -2.42515598e-01,
        -2.10627852e-01,  -1.68067036e-01,  -1.16998255e-01,
        -6.00146077e-02,   1.41713835e-16,   6.00146077e-02,
         1.16998255e-01,   1.68067036e-01,   2.10627852e-01,
         2.42515598e-01,   2.62120623e-01,   2.68503416e-01,
         2.61493651e-01,   2.41770948e-01,   2.10924929e-01,
         1.71492472e-01,   1.26970347e-01,   8.18017944e-02,
         4.13359551e-02,   1.17594741e-02,  -1.21088993e-02,
        -4.25710009e-02,  -8.42503848e-02,  -1.30774423e-01,
        -1.76633177e-01,  -2.17249842e-01,  -2.49022535e-01,
        -2.69338112e-01,  -2.76559151e-01,  -2.69985579e-01,
        -2.49792799e-01,  -2.16948511e-01,  -1.73110685e-01,
        -1.20509428e-01,  -6.18156999e-02,   1.47522868e-16,
         6.18156999e-02,   1.20509428e-01,   1.73110685e-01,
         2.16948511e-01,   2.49792799e-01,   2.69985579e-01,
         2.76559151e-01,   2.69338112e-01,   2.49022535e-01,
         2.17249842e-01,   1.76633177e-01,   1.30774423e-01,
         8.42503848e-02,   4.25710009e-02,   1.21088993e-02,
        -1.22268761e-02,  -4.29879892e-02,  -8.50771014e-02,
        -1.32058792e-01,  -1.78368831e-01,  -2.19385320e-01,
        -2.51470886e-01,  -2.71986636e-01,  -2.79279009e-01,
        -2.72641024e-01,  -2.52249801e-01,  -2.19082556e-01,
        -1.74813573e-01,  -1.21694906e-01,  -6.24238030e-02,
         1.51562347e-16,   6.24238030e-02,   1.21694906e-01,
         1.74813573e-01,   2.19082556e-01,   2.52249801e-01,
         2.72641024e-01,   2.79279009e-01,   2.71986636e-01,
         2.51470886e-01,   2.19385320e-01,   1.78368831e-01,
         1.32058792e-01,   8.50771014e-02,   4.29879892e-02,
         1.22268761e-02,  -1.21088993e-02,  -4.25710009e-02,
        -8.42503848e-02,  -1.30774423e-01,  -1.76633177e-01,
        -2.17249842e-01,  -2.49022535e-01,  -2.69338112e-01,
        -2.76559151e-01,  -2.69985579e-01,  -2.49792799e-01,
        -2.16948511e-01,  -1.73110685e-01,  -1.20509428e-01,
        -6.18156999e-02,   1.51617457e-16,   6.18156999e-02,
         1.20509428e-01,   1.73110685e-01,   2.16948511e-01,
         2.49792799e-01,   2.69985579e-01,   2.76559151e-01,
         2.69338112e-01,   2.49022535e-01,   2.17249842e-01,
         1.76633177e-01,   1.30774423e-01,   8.42503848e-02,
         4.25710009e-02,   1.21088993e-02,  -1.17594741e-02,
        -4.13359551e-02,  -8.18017944e-02,  -1.26970347e-01,
        -1.71492472e-01,  -2.10924929e-01,  -2.41770948e-01,
        -2.61493651e-01,  -2.68503416e-01,  -2.62120623e-01,
        -2.42515598e-01,  -2.10627852e-01,  -1.68067036e-01,
        -1.16998255e-01,  -6.00146077e-02,   1.48556062e-16,
         6.00146077e-02,   1.16998255e-01,   1.68067036e-01,
         2.10627852e-01,   2.42515598e-01,   2.62120623e-01,
         2.68503416e-01,   2.61493651e-01,   2.41770948e-01,
         2.10924929e-01,   1.71492472e-01,   1.26970347e-01,
         8.18017944e-02,   4.13359551e-02,   1.17594741e-02,
        -1.11919633e-02,  -3.93300731e-02,  -7.78249468e-02,
        -1.20792003e-01,  -1.63143262e-01,  -2.00652409e-01,
        -2.29993391e-01,  -2.48753194e-01,  -2.55419826e-01,
        -2.49346891e-01,  -2.30696462e-01,  -2.00362271e-01,
        -1.59875490e-01,  -1.11295651e-01,  -5.70893991e-02,
         1.42347445e-16,   5.70893991e-02,   1.11295651e-01,
         1.59875490e-01,   2.00362271e-01,   2.30696462e-01,
         2.49346891e-01,   2.55419826e-01,   2.48753194e-01,
         2.29993391e-01,   2.00652409e-01,   1.63143262e-01,
         1.20792003e-01,   7.78249468e-02,   3.93300731e-02,
         1.11919633e-02,  -1.04281322e-02,  -3.66302776e-02,
        -7.24723475e-02,  -1.12476326e-01,  -1.51905741e-01,
        -1.86826246e-01,  -2.14141561e-01,  -2.31605381e-01,
        -2.37810196e-01,  -2.32154324e-01,  -2.14788729e-01,
        -1.86545521e-01,  -1.48850252e-01,  -1.03620354e-01,
        -5.31522783e-02,   1.30915815e-16,   5.31522783e-02,
         1.03620354e-01,   1.48850252e-01,   1.86545521e-01,
         2.14788729e-01,   2.32154324e-01,   2.37810196e-01,
         2.31605381e-01,   2.14141561e-01,   1.86826246e-01,
         1.51905741e-01,   1.12476326e-01,   7.24723475e-02,
         3.66302776e-02,   1.04281322e-02,  -9.49739865e-03,
        -3.33405541e-02,  -6.59501700e-02,  -1.02343640e-01,
        -1.38212805e-01,  -1.69979081e-01,  -1.94826161e-01,
        -2.10710868e-01,  -2.16352998e-01,  -2.11205342e-01,
        -1.95405327e-01,  -1.69709977e-01,  -1.35416140e-01,
        -9.42681085e-02,  -4.83549513e-02,   1.17088950e-16,
         4.83549513e-02,   9.42681085e-02,   1.35416140e-01,
         1.69709977e-01,   1.95405327e-01,   2.11205342e-01,
         2.16352998e-01,   2.10710868e-01,   1.94826161e-01,
         1.69979081e-01,   1.38212805e-01,   1.02343640e-01,
         6.59501700e-02,   3.33405541e-02,   9.49739865e-03,
        -8.43579940e-03,  -2.95883187e-02,  -5.85110665e-02,
        -9.07864988e-02,  -1.22594983e-01,  -1.50763699e-01,
        -1.72795659e-01,  -1.86879355e-01,  -1.91879771e-01,
        -1.87311820e-01,  -1.73297480e-01,  -1.50508132e-01,
        -1.20093829e-01,  -8.36013887e-02,  -4.28833544e-02,
         1.00964797e-16,   4.28833544e-02,   8.36013887e-02,
         1.20093829e-01,   1.50508132e-01,   1.73297480e-01,
         1.87311820e-01,   1.91879771e-01,   1.86879355e-01,
         1.72795659e-01,   1.50763699e-01,   1.22594983e-01,
         9.07864988e-02,   5.85110665e-02,   2.95883187e-02,
         8.43579940e-03,  -7.28469039e-03,  -2.55198506e-02,
        -5.04451329e-02,  -7.82556691e-02,  -1.05661503e-01,
        -1.29929743e-01,  -1.48909598e-01,  -1.61040719e-01,
        -1.65345487e-01,  -1.61406153e-01,  -1.49327934e-01,
        -1.29689349e-01,  -1.03481294e-01,  -7.20364856e-02,
        -3.69510298e-02,   8.47361523e-17,   3.69510298e-02,
         7.20364856e-02,   1.03481294e-01,   1.29689349e-01,
         1.49327934e-01,   1.61406153e-01,   1.65345487e-01,
         1.61040719e-01,   1.48909598e-01,   1.29929743e-01,
         1.05661503e-01,   7.82556691e-02,   5.04451329e-02,
         2.55198506e-02,   7.28469039e-03,  -6.08920088e-03,
        -2.12948651e-02,  -4.20691750e-02,  -6.52434746e-02,
        -8.80778059e-02,  -1.08296066e-01,  -1.24106910e-01,
        -1.34210749e-01,  -1.37793366e-01,  -1.34506912e-01,
        -1.24439181e-01,  -1.08072262e-01,  -8.62317901e-02,
        -6.00281748e-02,  -3.07912618e-02,   6.95760910e-17,
         3.07912618e-02,   6.00281748e-02,   8.62317901e-02,
         1.08072262e-01,   1.24439181e-01,   1.34506912e-01,
         1.37793366e-01,   1.34210749e-01,   1.24106910e-01,
         1.08296066e-01,   8.80778059e-02,   6.52434746e-02,
         4.20691750e-02,   2.12948651e-02,   6.08920088e-03,
        -4.89646481e-03,  -1.70803150e-02,  -3.37144590e-02,
        -5.22647982e-02,  -7.05399012e-02,  -8.67191973e-02,
        -9.93697772e-02,  -1.07452059e-01,  -1.10314747e-01,
        -1.07679667e-01,  -9.96172159e-02,  -8.65132987e-02,
        -6.90287347e-02,  -4.80522321e-02,  -2.46481070e-02,
         5.53697079e-17,   2.46481070e-02,   4.80522321e-02,
         6.90287347e-02,   8.65132987e-02,   9.96172159e-02,
         1.07679667e-01,   1.10314747e-01,   1.07452059e-01,
         9.93697772e-02,   8.67191973e-02,   7.05399012e-02,
         5.22647982e-02,   3.37144590e-02,   1.70803150e-02,
         4.89646481e-03,  -3.75365382e-03,  -1.30435185e-02,
        -2.57131462e-02,  -3.98360589e-02,  -5.37459961e-02,
        -6.60584710e-02,  -7.56836724e-02,  -8.18308821e-02,
        -8.40047221e-02,  -8.19936781e-02,  -7.58514612e-02,
        -6.58718732e-02,  -5.25579332e-02,  -3.65860976e-02,
        -1.87664768e-02,   4.35099261e-17,   1.87664768e-02,
         3.65860976e-02,   5.25579332e-02,   6.58718732e-02,
         7.58514612e-02,   8.19936781e-02,   8.40047221e-02,
         8.18308821e-02,   7.56836724e-02,   6.60584710e-02,
         5.37459961e-02,   3.98360589e-02,   2.57131462e-02,
         1.30435185e-02,   3.75365382e-03,  -2.70583300e-03,
        -9.34471565e-03,  -1.83836395e-02,  -2.84525304e-02,
        -3.83659079e-02,  -4.71384321e-02,  -5.39942871e-02,
        -5.83704845e-02,  -5.99143054e-02,  -5.84752016e-02,
        -5.40915679e-02,  -4.69728647e-02,  -3.74776378e-02,
        -2.60880320e-02,  -1.33814446e-02,   3.32894106e-17,
         1.33814446e-02,   2.60880320e-02,   3.74776378e-02,
         4.69728647e-02,   5.40915679e-02,   5.84752016e-02,
         5.99143054e-02,   5.83704845e-02,   5.39942871e-02,
         4.71384321e-02,   3.83659079e-02,   2.84525304e-02,
         1.83836395e-02,   9.34471565e-03,   2.70583300e-03,
        -1.79364751e-03,  -6.12914850e-03,  -1.20150764e-02,
        -1.85643966e-02,  -2.50088067e-02,  -3.07092097e-02,
        -3.51621267e-02,  -3.80020679e-02,  -3.90000180e-02,
        -3.80582661e-02,  -3.52018666e-02,  -3.05670544e-02,
        -2.43869772e-02,  -1.69751381e-02,  -8.70697191e-03,
         2.46226734e-17,   8.70697191e-03,   1.69751381e-02,
         2.43869772e-02,   3.05670544e-02,   3.52018666e-02,
         3.80582661e-02,   3.90000180e-02,   3.80020679e-02,
         3.51621267e-02,   3.07092097e-02,   2.50088067e-02,
         1.85643966e-02,   1.20150764e-02,   6.12914850e-03,
         1.79364751e-03,  -1.05080590e-03,  -3.51873044e-03,
        -6.85121454e-03,  -1.05519835e-02,  -1.41898956e-02,
        -1.74055950e-02,  -1.99156241e-02,  -2.15140550e-02,
        -2.20718048e-02,  -2.15338189e-02,  -1.99143093e-02,
        -1.72902489e-02,  -1.37933352e-02,  -9.60064968e-03,
        -4.92425762e-03,   1.75564049e-17,   4.92425762e-03,
         9.60064968e-03,   1.37933352e-02,   1.72902489e-02,
         1.99143093e-02,   2.15338189e-02,   2.20718048e-02,
         2.15140550e-02,   1.99156241e-02,   1.74055950e-02,
         1.41898956e-02,   1.05519835e-02,   6.85121454e-03,
         3.51873044e-03,   1.05080590e-03,  -5.01188114e-04,
        -1.60328937e-03,  -3.07397267e-03,  -4.70066009e-03,
        -6.29669568e-03,  -7.70562932e-03,  -8.80367166e-03,
        -9.50073002e-03,  -9.74027607e-03,  -9.49820231e-03,
        -8.78078463e-03,  -7.62185563e-03,  -6.07929107e-03,
        -4.23091475e-03,  -2.16993121e-03,   1.14091710e-17,
         2.16993121e-03,   4.23091475e-03,   6.07929107e-03,
         7.62185563e-03,   8.78078463e-03,   9.49820231e-03,
         9.74027607e-03,   9.50073002e-03,   8.80367166e-03,
         7.70562932e-03,   6.29669568e-03,   4.70066009e-03,
         3.07397267e-03,   1.60328937e-03,   5.01188114e-04,
        -1.54880755e-04,  -4.31245730e-04,  -7.86994400e-04,
        -1.17600488e-03,  -1.55567296e-03,  -1.88955424e-03,
        -2.14851828e-03,  -2.31124185e-03,  -2.36430009e-03,
        -2.30196011e-03,  -2.12573176e-03,  -1.84371315e-03,
        -1.46976147e-03,  -1.02251809e-03,  -5.24316247e-04,
         5.59199719e-18,   5.24316247e-04,   1.02251809e-03,
         1.46976147e-03,   1.84371315e-03,   2.12573176e-03,
         2.30196011e-03,   2.36430009e-03,   2.31124185e-03,
         2.14851828e-03,   1.88955424e-03,   1.55567296e-03,
         1.17600488e-03,   7.86994400e-04,   4.31245730e-04,
         1.54880755e-04]
    grid_z0 = griddata(points, values, (grid_x,grid_y), method='linear')
     # print('grid_x,grid_y=',points)
    s=a.plot_surface(grid_x,grid_y,grid_z0)
    plt.show()

    hlist = [-log2(0.5), -log2(0.25), -log2(0.125), -log2(0.0625), -log2(0.03125)]
    error91 = [log2(0.0454144221912), log2(0.0111843305513), log2(0.00290980862588),log2(0.000729237444039), log2(0.000182430551479)]
    error92 = [log2(0.169787926651), log2(0.0382419715336), log2(0.0093030585368),log2(0.0023097703688), log2(0.000576444485371)]
    plt.plot(hlist, error91, label="inf_morm");
    plt.xlabel("log h");
    plt.ylabel("log2 (Discrete error)");
    plt.legend(['inf_norm'])
    plt.title('This is a plot of the error for the 9-point FD', color='red');
    plt.xlabel('-log2 (h)', color='red');
    plt.ylabel('log2(error)', color='red')
    plt.savefig('error91.png', transparent=True);
    plt.show()

    '''
    def vecsol(E,F,xs):
        G = zeros([(E + 1) * (F + 1), 1])
        G1 = array(G)
        if E==F==4:
            i = 0; j = E+2
            while(j<((E+2)+(F-1)) and i<(E-1)):
                a = xs[(i)*(1)]
                G1[j] = a
                i+=1;j+=1

            i = (E-1); j = (E+2)+(F+1)
            while (j<((E*F)-2) and i<(E+2)):
                a = xs[(i) * (1)]
                G1[j] = a
                i+=1; j+=1
            i = (E+2);j = E*F
            while (j < ((E*F)+F) and i < (E + (F+1))):
                a = xs[(i) * (1)]
                G1[j] = a
                i += 1;j += 1
            #print('G1 = ', G1)

        elif E>4:
            i = 0; i1 = 0
            while(i1<(E-1)):
                j = ((E+2)+(i1*(E+1)))
                while(i<((E-1)*(F-1)) and j < ((2*E)+1)+(i1*(E+1))):
                    a = xs[(i) * (1)]
                    G1[j] = a
                    i += 1; j+= 1
                i1 += 1
            #print('G1 = ', G1)
        return G1
    G1 = vecsol(E,F,xs)
    '''

    '''
    def Squares(E, F, h, k, r, s):
        z = sqrt(pow(h, 2) + pow(k, 2));
        d = E + 1;
        D = E + 2;
        l = len(r) - d;
        x10 = [];
        y10 = [];
        V = [];
        V1 = [];
        nodes = [];
        nummer = [];
        address = []
        for i in range(0, l - 1):
            for j in range(i + 1, i + 2):
                if (r[j] - r[i] <= h and s[i + d] - s[i] <= k and sqrt(
                                (r[i] - r[i + D]) ** 2 + (s[i] - s[i + D]) ** 2) <= z):
                    x = (r[i], r[j], r[i + D], r[i + d]);
                    V1.append(x);
                    x10.append(x)
                    y = (s[i], s[j], s[i + D], s[i + d]);
                    V1.append(y);
                    y10.append(y)
                    V.append([x, y]);  # V.append([x, y])
        # print('V = ', V); print('V1 = ', V1); print('x10 = ', x10);print('y10 = ', y10)
        print('')

        for d in range(0, len(r)):
            nodes.append([r[d], s[d]])
        # print('nodes = ', nodes)
        for d1 in range(0, len(nodes)):
            nummer.append(nodes.index(nodes[d1]))
        # print('nummer = ', nummer)
        print('')
        a = arange(1, (E + 1) * (F + 1) + 1, 1);  # print('a = ', a)
        ii = 0;
        w = []
        while (ii < (E * F) + (E - 1)):
            i = 0;
            j = 0;
            k = 0;
            q = 2
            while (i < q and j < q and k < q):
                w.append(a[ii + (k * i)])
                i += 1;
                j += 1;
                k += 1
            i1 = 0;
            j1 = 0;
            k1 = 0;
            q3 = 2;
            q1 = 5
            while (i1 < q3 and j1 < q3 and k1 < q3):
                w.append(a[(ii + k1 * i1) + q1])
                i1 += 1;
                j1 += 1;
                k1 += 1
            ii += 1
        # print('w = ', w)
        return (V, x10, y10, nodes)


    V, x10, y10, nodes = Squares(E, F, h, k, r, s)
    print('')


    '''
    '''
    f = plt.figure()
    a = f.gca(projection='3d')
    x = np.arange(-1, 1, 0.01)
    y = np.arange(-1, 1, 0.01)
    X, Z = np.meshgrid(x, y);u2 = xs[0]
    s = a.plot_surface(X,u2,Z, cmap=cm.jet)
    f.colorbar(s, shrink=0.5);
    plt.savefig('Aprox_solution.png', transparent=True)
    plt.title('This is a plot of the Approx solution', color='red')
    plt.xlabel('x-axis', color='red');plt.ylabel('y-axis', color='red')
    plt.show()
    '''

    '''
    def LagPol():
        xi, eta = symbols('xi eta')
        L1 = (1 / 4) * ((1 + xi) * (1 + eta))
        L2 = (1 / 4) * ((1 - xi) * (1 + eta))
        L3 = (1 / 4) * ((1 - xi) * (1 - eta))
        L4 = (1 / 4) * ((1 + xi) * (1 - eta))
        print('L1 = ', L1, 'L2 = ', L2, 'L3 = ', L3, 'L4 = ', L4 )
        return (L1, L2, L3, L4)
    L1, L2, L3, L4 = LagPol()
    print('')

    def trans(L1, L2, L3, L4, V):
        T = [];Z1 = [];Z2 = [];
        for j in range(0, (E + 1) * (F + 1)):
            i = 0
            T1 = L1 * V[0][0][i] + L2 * V[0][0][i + 1] + L3 * V[0][0][i + 2] + L4 * V[0][0][i + 3];
            Z1.append(T1)
            T2 = L1 * V[0][1][i] + L2 * V[0][1][i + 1] + L3 * V[0][1][i + 2] + L4 * V[0][1][i + 3];
            Z2.append(T2);
            T.append((Z1, Z2))
            # print('Z1 = ', Z1); print('Z2 = ', Z2)
        # print('T = ', T)
        return (T1, T2)
    T1, T2 = trans(L1, L2, L3, L4, V)
'''

    print('***************************************PROGRAM TERMINATED************************************************')


    def timestamp():
        import time
        t = time.time()
        print(time.ctime(t))
        return None


    def timestamp_test():
        import platform
        print('')
        print('TIMESTAMP_TEST:')
        print('  Python version: %s' % (platform.python_version()))
        print('  TIMESTAMP prints a timestamp of the current date and time.')
        print('')

        timestamp()
        #
        #  Terminate.
        #
        print('')
        print('TIMESTAMP_TEST:')
        print('  Normal end of execution.')
        return


    if (__name__ == '__main__'):
        timestamp_test()