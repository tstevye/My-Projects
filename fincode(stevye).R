
i <- 1
while (i < 3) {
  



print('Welcome to an R Program that solves basic Mathematical Finance Problems.')


menu(c("FOR THE SIMPLE INTEREST RATE MODULE", "FOR THE COMPOUND INTEREST RATE MODULE", "CONVERTING RATES MODULE", "ROOTS OF QUADRATIC & CUBIC MODULE", "DOLLAR WEIGHTED RATE MODULE", "TIME WEIGHTED RATE MODULE"), title="BELOW IS A LIST OF 6 MODULES FROM WHICH YOU CAN CHOSE DEPENDING ON WHICH ONE YOU WANT TO USE. FIRST ENTER 0 FROM YOUR KEYBOARD FOR THE SELECTION OPTION THAT FOLLOWS THIS LIST, THEN RUN THE PROGRAM ONE TIME. SECONDLY, ENTER THE NUMBER THAT CORRESPONDS TO THE MODULES OF YOUR CHOICE FROM YOUR KEYBOARD AND THEN RUN YOUR PROGRAM AGAIN:" )

D <- as.numeric (readline(prompt="ENTER THE NUMBER FOR SELECTED CHOICE? "))


if (D==1){
  print('Welcome to the Simplest Interest Rate Model Module.')
  
  menu(c("FUTURE VALUE", "PRESENT VALUE", "INTEREST", "INVESTMENT TIME", "RATE"), title=" WHAT DO YOU WANT TO CALCULATE? Choose a value for S1:")
  S1 <- as.numeric (readline(prompt="What is the value of S1? "))
  
  if (S1==1){
    print('YOU WANT TO CALCULATE THE FUTURE VALUE.')
    print('ENTER THE PRESENT VALUE, RATE AND TIME.')
    pv <-  as.integer(readline(prompt="Enter Present value: "))
    r <-  as.numeric(readline(prompt="Enter the rate: "))
    t <-  as.integer(readline(prompt="Enter the time: "))
    fv = pv*(1 + (r * t))
    print(paste("YOUR FUTURE VALUE IS, FV = ", fv))
  } else if (S1==2){
    print('YOU WANT TO CALCULATE THE PRESENT VALUE.')
    print('ENTER THE FUTURE VALUE, RATE AND TIME.')
    fv <-  as.integer(readline(prompt="Enter Future value: "))
    r <-  as.numeric(readline(prompt="Enter the rate: "))
    t <-  as.integer(readline(prompt="Enter the time: "))
    pv = fv/((1 + (r * t)))
    print(paste("YOUR PRESENT VALUE IS, PV = ", pv))
  } else if (S1==3){
    print('YOU WANT TO CALCULATE THE INTEREST.')
    print('ENTER THE PRESENT VALUE, RATE AND TIME OR PRESENT VALUE AND FUTURE VALUE.')
    pv <-  as.integer(readline(prompt="Enter Present value: "))
    fv <-  as.integer(readline(prompt="Enter Future value: "))
    r <-  as.numeric(readline(prompt="Enter the rate: "))
    t <-  as.integer(readline(prompt="Enter the time: "))
    I1 = pv*r*t
    I2 = fv - pv
    print(paste("YOUR INTEREST IS, I = ", I1))    # Interest using present value, rate and time
    print(paste("YOUR INTEREST IS, I = ", I2))    # Interest using future and present value
  } else if (S1==4){
    print('YOU WANT TO CALCULATE THE INVESTMENT TIME.')
    print('ENTER THE PRESENT VALUE, RATE AND FUTURE VALUE.')
    pv <-  as.integer(readline(prompt="Enter Present value: "))
    fv <-  as.integer(readline(prompt="Enter Future value: "))
    r <-  as.numeric(readline(prompt="Enter the rate: "))
    t = (fv-pv)/(pv*r)
    print(paste("YOUR INVESTMENT TIME IS, T = ", t))
  } else if (S1==5){
    print('YOU WANT TO CALCULATE THE INVESTMENT RATE.')
    print('ENTER THE PRESENT VALUE, TIME AND FUTURE VALUE.')
    pv <-  as.integer(readline(prompt="Enter Present value: "))
    fv <-  as.integer(readline(prompt="Enter Future value: "))
    t <-  as.numeric(readline(prompt="Enter the time: "))
    r = (fv-pv)/(pv*t)
    print(paste("YOUR INVESTMENT RATE IS, r = ", r))
  }
  
  
  #my.age <- as.integer(my.age)
  
} else if (D==2){
  print('Welcome to the Compound Interest Rate Model Module.')
  
  menu(c("FUTURE VALUE", "PRESENT VALUE", "INVESTMENT TIME", "RATE"), title=" WHAT DO YOU WANT TO CALCULATE? Choose a value for S1:")
  S2 <- as.numeric (readline(prompt="What is the value of S2? "))
  
  if (S2==1){
    print('YOU WANT TO CALCULATE THE FUTURE VALUE.')
    print('ENTER THE PRESENT VALUE, NOMINAL RATE AND TIME.')
    pv <-  as.integer(readline(prompt="Enter Present value: "))
    r1 <-  as.numeric(readline(prompt="Enter the nominal rate convertible monthly: "))
    t <-  as.integer(readline(prompt="Enter the time: "))
    r <- (1 + r1/12)**(12*t) - 1
    print(paste("YOUR EFFECTIVE INTEREST RATE IS, r = ", r))
    fv = pv*(1 + r)
    print(paste("YOUR FUTURE VALUE IS, FV = ", fv))
  } else if (S2==2){
    print('YOU WANT TO CALCULATE THE PRESENT VALUE.')
    print('ENTER THE FUTURE VALUE, RATE AND TIME.')
    fv <-  as.integer(readline(prompt="Enter Future value: "))
    r1 <-  as.numeric(readline(prompt="Enter the nominal rate convertible monthly: "))
    t <-  as.integer(readline(prompt="Enter the time: "))
    r <- (1 + r1/12)**(12*t) - 1
    print(paste("YOUR EFFECTIVE INTEREST RATE IS, r = ", r))
    pv = fv/(1+r)
    print(paste("YOUR PRESENT VALUE IS, PV = ", pv))
  }  else if (S2==3){
    print('YOU WANT TO CALCULATE THE INVESTMENT TIME.')
    print('ENTER THE PRESENT VALUE, RATE AND FUTURE VALUE.')
    pv <-  as.integer(readline(prompt="Enter Present value: "))
    fv <-  as.integer(readline(prompt="Enter Future value: "))
    r1 <-  as.numeric(readline(prompt="Enter the nominal rate convertible monthly: "))
    r <- (1 + r1/12)**(12*5) - 1
    print(paste("YOUR EFFECTIVE INTEREST RATE IS, r = ", r))
    t <- log(fv/pv)/log(1+r)
    print(paste("YOUR INVESTMENT TIME IS, T = ", t))
  } else if (S2==4){
    print('YOU WANT TO CALCULATE THE INVESTMENT RATE.')
    print('ENTER THE PRESENT VALUE, TIME AND FUTURE VALUE.')
    pv <-  as.integer(readline(prompt="Enter Present value: "))
    fv <-  as.integer(readline(prompt="Enter Future value: "))
    t <-  as.numeric(readline(prompt="Enter the time: "))
    r = (fv/pv)**(1/t) - 1
    print(paste("YOUR INVESTMENT RATE IS, r = ", r))
  }
  
  
  
  #my.age <- as.integer(my.age)
} else if(D==3){
  print('Welcome to the Module to Convert Between Effective Interest Rates, Nominal Interest Rates, Effective Discount Rate, Nominal Discount Rate and Force of Interest')
  
  menu(c("CONVERTING NOMINAL INTEREST RATE TO EFFECTIVE INTEREST RATE", "CONVERTING EFFECTIVE INTEREST RATE TO NOMUNAL INTEREST RATE", "CONVERTING EFFECTIVE INTEREST RATE TO EFFECTIVE DISCOUNT RATE", 
         "CONVERTING EFFECTIVE DISCOUNT RATE TO EFFECTIVE INTEREST RATE", "CONVERTING NOMINAL DISCOUNT RATE TO EFFECTIVE DISCOUNT RATE", "CONVERTING EFFECTIVE DISCOUNT RATE TO NOMINAL DISCOUNT RATE", 
         "CONVERTING NOMINAL INTEREST RATE TO NOMINAL DISCOUNT RATE", "CONVERTING NOMINAL DISCOUNT RATE TO NOMINAL INTEREST RATE", "CONVERTING NOMINAL DISCOUNT RATE TO EFFECTING INTEREST RATE", "CONVERTING EFFECTIVE INTEREST RATE TO NOMINAL DISCOUNT RATE", 
         "CONVERTING EFFECTIVE INTEREST RATE TO THE FORCE OF INTEREST", "CONVERTING THE FORCE OF INTEREST TO EFFECTIVE INTEREST RATE", "CONVERTING EFFECTIVE DISCOUNT RATE TO FORCE OF INTEREST", "CONVERTING FROM SIMPLE INTEREST RATE TO FORCE OF INTEREST", "CONVERTING FROM FORCE OF INTEREST TO SIMPLE INTEREST RATE"), title=" WHAT DO YOU WANT TO CALCULATE? Choose a value for S3:")
  S3 <- as.numeric (readline(prompt="What is the value of S3? "))
  
  if (S3==1){
    print('YOU WANT TO CONVERT FROM NOMINAL INTEREST RATE TO EFFECTIVE INETEREST RATE.')
    print('ENTER THE NOMINAL INTEREST RATE, INVESTMENT PERIOD AND COMPOUNDING FREQUENCY.')
    ni <-  as.numeric(readline(prompt="Enter the Nominal Interest Rate: "))
    t <-  as.numeric(readline(prompt="Enter the Investment Period: "))
    m <-  as.numeric(readline(prompt="Enter the Number of times the Interest is Compounded: "))
    i <- (1+(ni/m))**(m*t) - 1
    print(paste("YOUR EFFECTIVE INTEREST RATE FOR THIS INVESTMENT PERIOD IS, i = ", i))
  } else if (S3==2){
    print('YOU WANT TO CONVERT FROM EFFECTIVE INTEREST RATE TO NOMINAL INETEREST RATE.')
    print('ENTER THE NOMINAL INTEREST RATE, INVESTMENT PERIOD AND COMPOUNDING FREQUENCY.')
    e <-  as.numeric(readline(prompt="Enter the Effective Interest Rate: "))
    t <-  as.numeric(readline(prompt="Enter the Investment Period: "))
    m <-  as.numeric(readline(prompt="Enter the Number of times the Interest is Compounded: "))
    ni <- m*((1+e)**(t/m)-1)
    print(paste("YOUR NOMINAL INTEREST RATE FOR THIS INVESTMENT PERIOD IS, i = ", ni))
  }  else if (S3==3){
    print('YOU WANT TO CONVERT FROM EFFECTIVE INTEREST RATE TO EFFECTIVE DISCOUNT RATE.')
    print('ENTER THE EFFECTIVE INTEREST RATE.')
    e <-  as.numeric(readline(prompt="Enter the Effective Interest Rate: "))
    d <- e/(1+e)
    print(paste("YOUR EFFECTIVE DISCOUNT RATE FOR THIS INVESTMENT PERIOD IS, d = ", d))
  } else if (S3==4){
    print('YOU WANT TO CONVERT FROM EFFECTIVE DISCOUNT RATE TO EFFECTIVE INETEREST RATE.')
    print('ENTER THE EFFECTIVE DISCOUNT RATE.')
    d <-  as.numeric(readline(prompt="Enter the Effective Discount Rate: "))
    e <- d/(1-d)
    print(paste("YOUR EFFECTIVE INTEREST RATE FOR THIS INVESTMENT PERIOD IS, i = ", e))
  } else if (S3==5){
    print('YOU WANT TO CONVERT FROM NOMINAL DISCOUNT RATE TO EFFECTIVE DISCOUNT RATE.')
    print('ENTER THE NOMINAL DISCOUNT RATE, INVESTMENT PERIOD AND COMPOUNDING FREQUENCY.')
    d5 <-  as.numeric(readline(prompt="Enter the Nominal Discount Rate: "))
    t5 <-  as.numeric(readline(prompt="Enter the Investment Period: "))
    m5 <-  as.numeric(readline(prompt="Enter the Number of times the Interest is Compounded: "))
    ef5 <- 1-(1-(d5/m5))**(m5*t5)
    print(paste("YOUR EFFECTIVE DISCOUNT RATE FOR THIS INVESTMENT PERIOD IS, d = ", ef5))
  } else if (S3==6){
    print('YOU WANT TO CONVERT FROM EFFECTIVE DISCOUNT RATE TO NOMINAL DISCOUNT RATE.')
    print('ENTER THE EFFECTIVE DISCOUNT RATE, INVESTMENT PERIOD AND COMPOUNDING FREQUENCY.')
    d6 <-  as.numeric(readline(prompt="Enter the Effective Discount Rate: "))
    t6 <-  as.numeric(readline(prompt="Enter the Investment Period: "))
    m6 <-  as.numeric(readline(prompt="Enter the Number of times the Interest is Compounded: "))
    ef6 <- m6*(1-(1-d6)**(t6/m6))
    print(paste("YOUR NOMINAL DISCOUNT RATE FOR THIS INVESTMENT PERIOD IS, d = ", ef6))
  } else if (S3==7){
    print('YOU WANT TO CONVERT FROM NOMINAL INTEREST RATE TO NOMINAL DISCOUNT RATE.')
    print('ENTER THE NOMINAL INTEREST RATE, INVESTMENT PERIOD AND COMPOUNDING FREQUENCY.')
    i7 <-  as.numeric(readline(prompt="Enter the Nominal Interest Rate: "))
    t7 <-  as.numeric(readline(prompt="Enter the Investment Period: "))
    m7 <-  as.numeric(readline(prompt="Enter the Number of times the Interest is Compounded: "))
    d7 <- i7/(1+(i7/m7))
    print(paste("YOUR NOMINAL DISCOUNT RATE FOR THIS INVESTMENT PERIOD IS, d = ", d7))
  } else if (S3==8){
    print('YOU WANT TO CONVERT FROM NOMINAL DISCOUNT RATE TO NOMINAL INTEREST RATE.')
    print('ENTER THE NOMINAL DISCOUNT RATE, INVESTMENT PERIOD AND COMPOUNDING FREQUENCY.')
    d8 <-  as.numeric(readline(prompt="Enter the Nominal Discount Rate: "))
    t8 <-  as.numeric(readline(prompt="Enter the Investment Period: "))
    m8 <-  as.numeric(readline(prompt="Enter the Number of times the Interest is Compounded: "))
    i8 <- d8/(1-(d8/m8))
    print(paste("YOUR NOMINAL INTEREST RATE FOR THIS INVESTMENT PERIOD IS, i = ", i8))
  } else if (S3==9){
    print('YOU WANT TO CONVERT FROM NOMINAL DISCOUNT RATE TO EFFECTIVE INTEREST RATE.')
    print('ENTER THE NOMINAL DISCOUNT RATE, INVESTMENT PERIOD AND COMPOUNDING FREQUENCY.')
    d9 <-  as.numeric(readline(prompt="Enter the Nominal Discount Rate: "))
    t9 <-  as.numeric(readline(prompt="Enter the Investment Period: "))
    m9 <-  as.numeric(readline(prompt="Enter the Number of times the Interest is Compounded: "))
    i9 <- 1/(1-(d9/m9))**(m9*t9) - 1
    print(paste("YOUR EFFECTIVE INTEREST RATE FOR THIS INVESTMENT PERIOD IS, i = ", i9))
  } else if (S3==10){
    print('YOU WANT TO CONVERT FROM EFFECTIVE INTEREST RATE TO NOMINAL DISCOUNT RATE.')
    print('ENTER THE EFFECTIVE INTEREST RATE, INVESTMENT PERIOD AND COMPOUNDING FREQUENCY.')
    i10 <-  as.numeric(readline(prompt="Enter the Effective Interest Rate: "))
    t10 <-  as.numeric(readline(prompt="Enter the Investment Period: "))
    m10 <-  as.numeric(readline(prompt="Enter the Number of times the Interest is Compounded: "))
    d10 <- m10*(1-(1/(1+i10)**(1/m10*t10)))
    print(paste("YOUR NOMINAL DISCOUNT RATE FOR THIS INVESTMENT PERIOD IS, d = ", d10))
  } else if (S3==11){
    print('YOU WANT TO CONVERT FROM EFFECTIVE INTEREST RATE TO FORCE OF INTEREST.')
    print('ENTER THE EFFECTIVE INTEREST RATE.')
    i11 <-  as.numeric(readline(prompt="Enter the Effective Interest Rate: "))
    f11 <- ln1p(1+i11)
    print(paste("YOUR FORCE OF INTEREST FOR THIS INVESTMENT PERIOD IS, f = ", f11))
  } else if (S3==12){
    print('YOU WANT TO CONVERT FROM FORCE OF INTEREST TO EFFECTIVE INTEREST RATE.')
    print('ENTER THE FORCE OF INTEREST.')
    f12 <-  as.numeric(readline(prompt="Enter the Force of Interest: "))
    i12 <- exp(f12) - 1
    print(paste("YOUR EFFECTIVE INTEREST RATE FOR THIS INVESTMENT PERIOD IS, i = ", i12))
  } else if (S3==13){
    print('YOU WANT TO CONVERT FROM EFFECTIVE DISCOUNT RATE TO FORCE OF INTEREST.')
    print('ENTER THE EFFECTIVE DISCOUNT RATE, INVESTMENT PERIOD.')
    d13 <-  as.numeric(readline(prompt="Enter the Effective Discount Rate: "))
    t13 <-  as.numeric(readline(prompt="Enter the Investment Period: "))
    f13 <- d13/(1-(d13*t13))
    print(paste("YOUR FORCE OF INTEREST FOR THIS INVESTMENT PERIOD IS, f = ", f13))
  } else if (S3==14){
    print('YOU WANT TO CONVERT FROM SIMPLE INTEREST RATE TO FORCE OF INTEREST.')
    print('ENTER THE SIMPLE INTEREST RATE, INVESTMENT PERIOD.')
    r14 <-  as.numeric(readline(prompt="Enter the Simple Interest Rate: "))
    t14 <-  as.numeric(readline(prompt="Enter the Investment Period: "))
    d14 <- r14/(1+(r14*t14))
    print(paste("YOUR FORCE OF INTEREST FOR THIS INVESTMENT PERIOD IS, f = ", d14))
  } else if (S3==15){
    print('YOU WANT TO CONVERT FROM FORCE OF INTEREST RATE TO SIMPLE INTEREST RATE.')
    print('ENTER THE FORCE OF INETEREST, INVESTMENT PERIOD.')
    f15 <-  as.numeric(readline(prompt="Enter the Force of Interest: "))
    t15 <-  as.numeric(readline(prompt="Enter the Investment Period: "))
    s15 <- f15/(1-(t15*f15))
    print(paste("YOUR SIMPLE INTEREST RATE FOR THIS INVESTMENT PERIOD IS, S = ", s15))
  }
  
  
  
  print('')
} else if(D==4){
  print('Welcome to the Module that Calculates the Roots of Quadratic and Cubic Equation')
  
  ky <-  as.numeric(readline(prompt="ENTER THE HIGHEST ORDER OF YOUR EQUATION: "))
  
  if (ky == 2){
    print(paste("!!!!!!!!!!!!!!!!!!!!!!!YOU WANT TO SOLVE A QUADRATIC EQUATION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"))
    kx <-  as.numeric(readline(prompt="Enter the number of coefficients you wish to consider: "))
    print('YOU WOULD HAVE TO ENTER THESE COEFFICIENTS IN THE ORDER a, b, c AND d.')
    cof = numeric()
    i <- 1
    while (i < kx+1) {
      ps <-  as.numeric(readline(prompt=": Enter the coefficient:"))
      cof <- append(cof,ps)
      
      i = i+1
    }
    print(paste("ALL YOUR COEFFICIENTS ARE:"))
    print(cof)
    
    gl <- (cof[[2]])**2-4*(cof[[1]])*(cof[[3]])
    if (gl<0){
      print(paste("!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!!!!!"))
      print(paste("!!!!!!!!!!!!!!!!!!!!!NO REAL SOLUTION TO THIS QUADRATIC EQUATION EXIST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"))
      
    } else if (gl >= 0){
      gb1 <- -1*(cof[[2]])+sqrt((cof[[2]])**2-4*(cof[[1]])*(cof[[3]]))
      gb2 <- gb1/(2*(cof[[1]]))
      gb3 <- -1*(cof[[2]])-sqrt((cof[[2]])**2-4*(cof[[1]])*(cof[[3]]))
      gb4 <- gb3/(2*(cof[[1]]))
      
      if (gb2<=0){
        print(paste("!!!!!!!!!!!!!!!!!!!!!!!!THIS ROOT IS UNDEFINED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"))
        
      } else if (gb2 > 0){
        print(paste("THE REQUIRED ROOT TO YOUR QUADTRATIC EQUATION IS, x1 = ", gb2))
        I1 <- gb2-1
      }
      
      if (gb4<=0){
        print(paste("!!!!!!!!!!!!!!!!!!!!!!!!THIS ROOT IS UNDEFINED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"))
        
      } else if (gb4 > 0){
        print(paste("THE REQUIRED ROOT TO YOUR QUADTRATIC EQUATION IS, x2 = ", gb4))
        I2 <- gb4-1
      }
      
      if (I2<0){
        print(paste("!!!!!!!!!!!!!!!!!!! INVALID INTERNAL RATE OF RETURN !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"))
      } else if (I2 >= 0){
        print(paste("THE INTERNAL RATE OF RETURN TO THIS TRANSACTION IS, i = ", I2))
      }
      if (I1<0){
        print(paste("!!!!!!!!!!!!!!!!!!! INVALID INTERNAL RATE OF RETURN !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"))
      } else if (I1 >= 0){
        print(paste("THE INTERNAL RATE OF RETURN TO THIS TRANSACTION IS, i = ", I1))
      }
      
      
    } 
    
  } else if (ky == 3){
    print(paste("!!!!!!!!!!!!!!!!!!!!!!!YOU WANT TO SOLVE A CUBIC EQUATION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"))
    kx <-  as.numeric(readline(prompt="Enter the number of coefficients you wish to consider: "))
    print('YOU WOULD HAVE TO ENTER THESE COEFFICIENTS IN THE ORDER a, b, c AND d.')
    print('NOW, TYPE THE WORD "cubic()" AT THE TERMINAL(OR CONSOLE) .')
    cubic <- function()
      
    {
      a <- readline(prompt='ENTER A NONZERO VALUE OF a: ')
      
      if(a==0){
        
        message(paste('!!!!!!Warning!!!!!!WARNING!!!!!!! HIGHEST DEGREE POLY COEFFICIENT CANNOT BE ZERO !!!!!!!!!!!!!'))
        
      }
      else{
        b <- readline(prompt='ENTER A NONZERO VALUE OF b: ')
        c <- readline(prompt='ENTER A NONZERO VALUE OF c: ')
        d <- readline(prompt='ENTER A NONZERO VALUE OF d: ')}
      a <- as.numeric(unlist(strsplit(a, ",")))
      b <- as.numeric(unlist(strsplit(b, ",")))
      c <- as.numeric(unlist(strsplit(c, ",")))
      d <- as.numeric(unlist(strsplit(d, ",")))
      denom = a
      a = b/denom
      b = c/denom
      c = d/denom
      
      pietwo = 2.0*pi
      piefour = 4.0*pi
      p = a/3.0
      q = (3*b-a*a)/(9.0)
      qw = q*q*q
      r = (9*a*b-27*c-2*a*a*a)/54.0
      rw = r*r
      f = qw +  rw
      if(f<0.0){
        message(paste('Three unequal roots'))
        theta = acos((r/(sqrt(-1*qw))))
        qs = sqrt(-1*q)
        root1 = 2.0*qs*cos(theta/3.0)-p
        root2 = 2.0*qs*cos((theta + pietwo)/3.0)-p
        root3 = 2.0*qs*cos((theta + piefour)/3.0)-p
      }
      
      else if(f>0.0){
        message(paste('One real root'))
        dsq = sqrt(f)
        s = (r+dsq)^(1/3)
        t = (r-dsq)^(1/3)
        root1 = s + t-p
        root2 = print('Na')
        root3 = print('Na')
      }
      
      else
        
      {
        message(paste('Three real roots, at least two equal'))
        rcb= (r)^(1/3)
        root1 = 2.0*rcb-p
        root2 = root3 = rcb-p
      }
      
      print(paste("THE REQUIRED ROOT TO YOUR QUADTRATIC EQUATION ARE = ", root1, root2, root3))
      #list(root1=root1, root2=root2, root3=root3)
      if (root1<=0){
        print(paste("!!!!!!!!!!!!!!!!!!!!!!!!THIS ROOT IS UNDEFINED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"))
        
      } else if (root1 > 0){
        #print(paste("THE REQUIRED ROOT TO YOUR QUADTRATIC EQUATION IS, x1 = ", root1))
        r1 <- root1-1
      }
      
      if (root2<=0){
        print(paste("!!!!!!!!!!!!!!!!!!!!!!!!THIS ROOT IS UNDEFINED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"))
        
      } else if (root2 > 0){
        #print(paste("THE REQUIRED ROOT TO YOUR QUADTRATIC EQUATION IS, x2 = ", root2))
        r2 <- root2-1
      } 
      
      if (root3<=0){
        print(paste("!!!!!!!!!!!!!!!!!!!!!!!!THIS ROOT IS UNDEFINED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"))
        
      } else if (root3 > 0){
        #print(paste("THE REQUIRED ROOT TO YOUR QUADTRATIC EQUATION IS, x2 = ", root3))
        r3 <- root3-1
      } 
      
      if (r3<0){
        print(paste("!!!!!!!!!!!!!!!!!!! INVALID INTERNAL RATE OF RETURN !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"))
      } else if (r3 >= 0){
        print(paste("THE INTERNAL RATE OF RETURN TO THIS TRANSACTION IS, i = ", r3))
      }
      if (r2<0){
        print(paste("!!!!!!!!!!!!!!!!!!! INVALID INTERNAL RATE OF RETURN !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"))
      } else if (r2 >= 0){
        print(paste("THE INTERNAL RATE OF RETURN TO THIS TRANSACTION IS, i = ", r2))
      }
      if (r1<0){
        print(paste("!!!!!!!!!!!!!!!!!!! INVALID INTERNAL RATE OF RETURN !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"))
      } else if (r1 >= 0){
        print(paste("THE INTERNAL RATE OF RETURN TO THIS TRANSACTION IS, i = ", r1))
      }
      
    }
    
    
    
    
  }
  
  
  
  
  
  
  print('')
} else if(D==5){
  print('Welcome to the Dollar Weighted Yield Rate Module')
  
  print('YOU WOULD HAVE TO ENTER THE NUMBER OF CONTRIBUTIONS, NUMBER OF WITHDRAWALS, INITIAL BALANCE, FINAL BALANCE.')
  Y1 <-  as.numeric(readline(prompt="Enter the year of the initial investment account balance: "))
  M1 <-  as.numeric(readline(prompt="Enter the month of the initial investment account balance: "))
  A <- as.numeric(readline(prompt="Enter Your account initial balance: "))
  Y2 <-  as.numeric(readline(prompt="Enter the year of the final investment account balance: "))
  M2 <-  as.numeric(readline(prompt="Enter the month of the final investment account balance: "))
  B <- as.numeric(readline(prompt="Enter Your account Final balance: "))
  r = M2-M1
  s = Y2-Y1
  
  if (s==0){
    p = 12
    print(paste("YOUR LENGTH OF INVESTMENT IS, L = ", p, "Months"))
    
  } else if (r!=0){
    p = ((Y2-Y1)+1)*12 - (abs(M2-M1))
    print(paste("YOUR LENGTH OF INVESTMENT IS, L = ", p, "Months"))
    
  } else if (r==0){
    p = ((Y2-Y1)+1)*12 - 12
    print(paste("YOUR LENGTH OF INVESTMENT IS, L = ", p, "Months"))
    
  }
  
  
  c1 <-  as.numeric(readline(prompt="Enter the number of contributions you wish to consider: "))
  w1 <-  as.numeric(readline(prompt="Enter the number of withdrawals you wish to consider: "))
  
  contr = numeric()
  witd = numeric()
  mcontr = numeric()
  mwitd = numeric()
  xx = numeric()
  yy = numeric()
  i <- 1
  while (i < c1+1) {
    x1 <-  as.numeric(readline(prompt=": Enter the contribution:"))
    contr <- append(contr,x1)
    x22 <-  as.numeric(readline(prompt=": Enter the year of this contribution:"))
    x2 <-  as.numeric(readline(prompt=": Enter the month of this contribution:"))
    mcontr <- append(mcontr,x2)
    
    if (x22==Y1){
      zz = x1*(1-((x2-1)/p))
      xx <- append(xx,zz)
      
    } else if (x22>Y1){
      pz1 <- ((x22-Y1)*12)+(y2-1)
      pz1
      zz = y1*(1-(pz1/p))
      xx <- append(xx,zz)
    }
    #print(x1)
    i = i+1
  }
  print(paste("ALL YOUR CONTRIBUTIONS ARE/IS:"))
  print(contr)
  print(paste("AND THE CORRESPONDING MONTHS OF THESE CONTRIBUTIONS ARE/IS:"))
  print(mcontr)
  #print(paste("THESE CONTRIBUTIONS SUBJECT TO THEIR CORRESPONDING MONTHS ARE:"))
  #print(xx)
  
  j <- 1
  while (j < w1+1) {
    y1 <-  as.numeric(readline(prompt=": Enter the withdrawals:"))
    witd <- append(witd,y1)
    y22 <-  as.numeric(readline(prompt=": Enter the year of this withdrawal:"))
    y2 <-  as.numeric(readline(prompt=": Enter the month of this withdrawal:"))
    mwitd <- append(mwitd,y2)
    
    if (y22==Y1){
      zz1 = y1*(1-((y2-1)/p))
      yy <- append(yy,zz1)
      
    } else if (y22>Y1){
      pz <- ((y22-Y1)*12)+(y2-1)
      pz
      zz1 = y1*(1-(pz/p))
      yy <- append(yy,zz1)
    }
    
    j = j+1
  }
  print(paste("ALL YOUR WITHDRAWALS ARE/IS:"))
  print(witd)
  print(paste("AND THE CORRESPONDING MONTHS OF THESE WITHDRAWALS ARE/IS:"))
  print(mwitd)
  #print(paste("THESE withdrawals SUBJECT TO THEIR CORRESPONDING MONTHS ARE:"))
  #print(yy)
  
  b1 <- sum(contr, na.rm = FALSE)
  b2 <- sum(witd, na.rm = FALSE)
  z = b1 - b2
  I = B-A-z
  bB1 <- sum(xx, na.rm = FALSE)
  bB2 <- sum(yy, na.rm = FALSE)
  F <- bB1 - bB2
  j <- I/(A+F)
  print(paste("YOUR APPROXIMATE ANNUAL DOLLAR-WEIFGTED YIELD RATE IS, j = ", j))
  
  
  print('')
} else if(D==6){
  print('Welcome to the Time Weighted Rate Module.')
  
  print('YOU WOULD HAVE TO ENTER THE NUMBER OF CONTRIBUTIONS, NUMBER OF WITHDRAWALS, INITIAL BALANCE, FINAL BALANCE.')
  
  d1 <-  as.numeric(readline(prompt="Enter the number of account balances in your entire investment period: "))
  
  bal = numeric()
  i <- 1
  while (i < d1+1) {
    x18 <-  as.numeric(readline(prompt=": Enter the current account balance:"))
    bal <- append(bal,x18)
    
    i = i+1
  }
  print(paste("ALL YOUR BALANCES ARE/IS:"))
  print(bal)
  
  Y11 <-  as.numeric(readline(prompt="Enter the year of the initial(very first) investment account balance: "))
  Y12 <-  as.numeric(readline(prompt="Enter the year of the final(last) investment account balance: "))
  k = Y12 - Y11
  k
  
  cont = numeric()
  cont <- append(cont,-bal[[1]])
  i <- 1
  while (i < d1-1) {
    xh <-  as.numeric(readline(prompt=": Enter the contributions and/or withdrawals in the appropriate order :"))
    cont <- append(cont,xh)
    
    i = i+1
  }
  cont <- append(cont,-bal[[d1]])
  print(paste("ALL YOUR CONTRIBUTIONS ARE/IS:"))
  print(cont)
  
  difr = numeric()
  difr1 = numeric()
  jh <- bal + cont
  difr <- append(difr,jh)
  print(paste("YOUR BALANCES AFTER CONTRIBUTIONS AND WITHDRAWALS ARE/IS:"))
  print(difr)
  
  jK <- bal[[2]]/bal[[1]]
  difr1 <- append(difr1,jK)
  i <- 1
  while (i < d1-1) {
    gk <- bal[[i+2]]/difr[[i+1]]
    difr1 <- append(difr1,gk)
    
    i = i+1
  }
  #print(paste("YOUR QUOTIENT VECTOR IS:"))
  #print(difr1)
  
  if (k!=0){
    mol <- (prod(difr1, na.rm = FALSE))**(1/k) -1
    
  } else if (k==0){
    mol <- prod(difr1, na.rm = FALSE) - 1
  }
  print(paste("YOUR ANNUAL TIME-WEIFGTED RATE OF RETURN IS, j = ", mol))
  
  print('')
}

i = i+1
}

