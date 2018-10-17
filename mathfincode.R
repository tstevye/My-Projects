
i <- 1
while (i < 3) {

print('Welcome to an R Program that solves basic Mathematical Finance Problems on Annuities.')

menu(c("PAYMENT IS MADE AT THE END OF EACH INTEREST PERIOD(ANNUIY IMMEDIATE)", "PAYMENT IS MADE AT THE BEGINING OF EACH INTEREST PERIOD(ANNUITY DUE)"), title="BELOW ARE TWO OPTIONS YOU SHOULD CHOOSE TO PROCEED WITH YOUR COMPUTATION. FIRST ENTER 0 FROM YOUR KEYBOARD FOR THE SELECTION OPTION THAT FOLLOWS THIS LIST, THEN RUN THE PROGRAM ONE TIME. SECONDLY, ENTER THE NUMBER THAT CORRESPONDS TO THE OPTION OF YOUR CHOICE FROM YOUR KEYBOARD AND THEN RUN YOUR PROGRAM AGAIN:" )
print('!!!!!!!!!!!!WARNING!!!!!!!!!! MAKE SURE YOUR CHOICE IS EITHER 1 OR 2 !!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!')
D1 <- as.numeric (readline(prompt="ENTER THE NUMBER FOR THE SELECTED CHOICE? "))


if (D1==1){
  print('Welcome to the Annuity Immediate Module.')
  menu(c("LEVEL ANNUITIES", "NONLEVEL ANNUITIES"), title="BELOW ARE TWO OPTIONS YOU SHOULD CHOOSE TO PROCEED WITH YOUR COMPUTATION. FIRST ENTER 0 FROM YOUR KEYBOARD FOR THE SELECTION OPTION THAT FOLLOWS THIS LIST, THEN RUN THE PROGRAM ONE TIME. SECONDLY, ENTER THE NUMBER THAT CORRESPONDS TO THE OPTION OF YOUR CHOICE FROM YOUR KEYBOARD AND THEN RUN YOUR PROGRAM AGAIN:" )
  print('!!!!!!!!!!!!WARNING!!!!!!!!!! MAKE SURE YOUR CHOICE IS EITHER 1 OR 2 !!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!')
  D <- as.numeric (readline(prompt="ENTER THE NUMBER FOR THE SELECTED CHOICE? "))
  
  if (D==1){
  
      print('Welcome to the Annuity Immediate Module with Level Payments.')
      menu(c("PRESENT VALUE", "FUTURE VALUE", "PAYMENT", "INVESTMENT TIME", "RATE"), title=" WHAT DO YOU WANT TO CALCULATE? Choose a value for S1:")
      S1 <- as.numeric (readline(prompt="Enter the value of S1: "))
      
      if (S1==1){
        print('YOU WANT TO CALCULATE THE PRESENT VALUE OF YOUR ANNUITY IMMEDIATE.')
        pmt <-  as.integer(readline(prompt="Enter the Payment Amount/Contribution: "))
        r <-  as.numeric(readline(prompt="Enter the Effective Interest Rate per Interest Period: "))
        t <-  as.integer(readline(prompt="Enter the Number of Interest Periods: "))
        pv = pmt*((1-(1+r)^(-t))/r)
        print(paste("YOUR PRESENT VALUE IS, PV = ", pv))
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      } else if (S1==2){
        print('YOU WANT TO CALCULATE THE FUTERE VALUE OF YOUR ANNUITY IMMEDIATE.')
        pmt1 <-  as.integer(readline(prompt="Enter the Payment Amount/Contribution: "))
        r1 <-  as.numeric(readline(prompt="Enter the Effective Interest Rate per Interest Period: "))
        t1 <-  as.integer(readline(prompt="Enter the Number of Interest Periods: "))
        fv = pmt1*(((1+r1)^(t1)-1)/r1)
        print(paste("YOUR FUTURE VALUE IS, FV = ", fv))
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      } else if (S1==3){
            menu(c("IF YOU WANT TO USE PRESENT VALUE", "IF YOU WANT TO USE FUTURE VALUE"), title="Choose a value for S1 to proceed with the calculation of Payment/Contribution, t:")
            SS2 <- as.numeric (readline(prompt="Enter the value of S1: "))
            
            if (SS2==1){
              print('YOU WANT TO CALCULATE THE Payment/Contribution OF YOUR ANNUITY IMMEDIATE USING THE PRESENT VALUE.')
              rr <-  as.numeric(readline(prompt="Enter the Effective Interest Rate per Interest Period: "))
              tt <-  as.integer(readline(prompt="Enter the Number of Interest Periods: "))
              ppv = (1-(1+rr)^(-tt))/rr
              ppv3 <-  as.integer(readline(prompt="Enter the Amount you wish to get(OR Present Value): "))
              ppmt = ppv3/ppv
              print(paste("YOUR PAYMENT/CONTRIBUTION IS, PMT = ", ppmt))
              print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
              
            } else if (SS2==2){
              print('YOU WANT TO CALCULATE THE Payment/Contribution OF YOUR ANNUITY IMMEDIATE USING THE FUTURE VALUE.')
              rr1 <-  as.numeric(readline(prompt="Enter the Rate per Interest Period: "))
              tt1 <-  as.integer(readline(prompt="Enter the Number of Interest Periods: "))
              ffv = ((1+rr1)^(tt1)-1)/rr1
              ppv3 <-  as.integer(readline(prompt="Enter the Amount you wish to Accoumulate (Futur Value): "))
              ppmt = ppv3/ffv
              print(paste("YOUR PAYMENT/CONTRIBUTION IS, PMT = ", ppmt))
              print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            }
        
      } else if (S1==4){
        menu(c("IF YOU WANT TO USE PRESENT VALUE", "IF YOU WANT TO USE FUTURE VALUE"), title="Choose a value for S1 to proceed with the calculation of your investment time, t:")
        S2 <- as.numeric (readline(prompt="Enter the value of S1: "))
        
            if (S2==1){
              print('YOU WANT TO CALCULATE THE INVESTMENT TIME OF YOUR ANNUITY IMMEDIATE USING THE PRESENT VALUE.')
              pv3 <-  as.integer(readline(prompt="Enter Present value: "))
              pmt3 <-  as.integer(readline(prompt="Enter the Payment/Conribution: "))
              r3 <-  as.numeric(readline(prompt="Enter the Effective Interest rate: "))
              t3 = -1*(log(1-(r3*pv3/pmt3)) /log(1+r3))
              print(paste("YOUR INVESTMENT TIME IS, T = ", t3))
              print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
              
            } else if (S2==2){
              print('YOU WANT TO CALCULATE THE INVESTMENT TIME OF YOUR ANNUITY IMMEDIATE USING THE FUTURE VALUE.')
              fv3 <-  as.integer(readline(prompt="Enter future value: "))
              pmt3 <-  as.integer(readline(prompt="Enter the Payment/Conribution: "))
              r3 <-  as.numeric(readline(prompt="Enter the Effective Interest rate: "))
              t3 = log(1+(r3*fv3/pmt3))/log(1+r3)
              print(paste("YOUR INVESTMENT TIME IS, T = ", t3))
              print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            }
    
      } else if (S1==5){
        print('YOU WANT TO CALCULATE THE EFFECTIVE INTEREST RATE.')
        print('SORRY THIS MODULE IS NOT CURRENTLY AVAILABLE')
        
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      }
  } else if (D==2){
      print('Welcome to the Annuity Immediate Module with NonLevel Payments.')
      menu(c("GEOMETRIC PROGRESSION", "ARITHMETIC PROGRESSION"), title=" WHAT DO YOU WANT TO CALCULATE? Choose a value for S1:")
      K1 <- as.numeric (readline(prompt="Enter the value of S1: "))
          if (K1==1){
          
              menu(c("PRESENT VALUE", "FUTURE VALUE"), title=" WHAT DO YOU WANT TO CALCULATE? Choose a value for S1:")
              SK1 <- as.numeric (readline(prompt="Enter the value of S1: "))
              
              if (SK1==1){
                print('YOU WANT TO CALCULATE THE PRESENT VALUE OF YOUR NONLEVEL ANNUITY WITH GEOMETRIC PROGRESSION.')
                npmt <-  as.integer(readline(prompt="Enter the Payment Amount/Contribution: "))
                nr <-  as.numeric(readline(prompt="Enter the effective interest Rate per Interest Period: "))
                ng <-  as.numeric(readline(prompt="Enter the Growth Rate: "))
                nt <-  as.integer(readline(prompt="Enter the Number of Interest Periods: "))
                    if (nr!=ng){
                      jz <- ((1+ng)/(1+nr))^(nt)
                      jz1 <- ((1+nr)-(1+ng))
                      npv = npmt*((1-jz)/jz1)
                      print(paste("YOUR PRESENT VALUE IS, PV = ", npv))
                      print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    }else if(nr==ng){
                      npv = (nt*npmt)/(1+nr)
                      print(paste("YOUR PRESENT VALUE IS, PV = ", npv))
                      print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    }

              } else if (SK1==2){
                print('YOU WANT TO CALCULATE THE FUTERE VALUE OF YOUR NONLEVEL ANNUITY WITH GEOMETRIC PROGRESSION.')
                npmt1 <-  as.integer(readline(prompt="Enter the Payment Amount/Contribution: "))
                nr1 <-  as.numeric(readline(prompt="Enter the effective interest Rate per Interest Period: "))
                ng1 <-  as.numeric(readline(prompt="Enter the Growth Rate: "))
                nt1 <-  as.integer(readline(prompt="Enter the Number of Interest Periods: "))

                nfv = npmt1*(((1+nr1)^(nt1) - (1+ng1)^(nt1))/(nr1-ng1))
                print(paste("YOUR FUTURE VALUE IS, FV = ", nfv))
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
              }
          
          
          }else if (K1==2){
            
              menu(c("FOR INCREASING ARITHMETIC PROGRESSION", "FOR DECREASING ARITHMETIC PROGRESSION"), title=" WHAT DO YOU WANT TO CALCULATE? Choose a value for S1:")
              SSK1 <- as.numeric (readline(prompt="What is the value of S1? "))
              if (SSK1==1){
                
                  menu(c("PRESENT VALUE", "FUTURE VALUE"), title=" WHAT DO YOU WANT TO CALCULATE? Choose a value for S1:")
                  SK1 <- as.numeric (readline(prompt="Enter the value of S1: "))
                  
                  if (SK1==1){
                    print('YOU WANT TO CALCULATE THE PRESENT VALUE OF YOUR NONLEVEL ANNUITY WITH INCREASING ARITHMETIC PROGRESSION.')
                    pmTc <-  as.integer(readline(prompt="Enter the Payment Amount/Contribution: "))
                    qmTc <-  as.integer(readline(prompt="Enter the  Amount/Contribution of increase: "))
                    rc <-  as.numeric(readline(prompt="Enter the Effective Interest Rate per Interest Period: "))
                    tc <-  as.integer(readline(prompt="Enter the Number of Interest Periods: "))
                    pvc = pmTc*((1-(1+rc)^(-tc))/rc) + (qmTc/rc)*(((1-(1+rc)^(-tc))/rc)-tc*((1+rc)^(-tc)))
                    print(paste("YOUR PRESENT VALUE IS, PV = ", pvc))
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                  } else if (SK1==2){
                    print('YOU WANT TO CALCULATE THE FUTERE VALUE OF YOUR NONLEVEL ANNUITY WITH INCREASING ARITHMETIC PROGRESSION.')
                    pmTc1 <-  as.integer(readline(prompt="Enter the Payment Amount/Contribution: "))
                    qmTc <-  as.integer(readline(prompt="Enter the  Amount/Contribution of increase: "))
                    rc1 <-  as.numeric(readline(prompt="Enter the Rate per Interest Period: "))
                    tc1 <-  as.integer(readline(prompt="Enter the Number of Interest Periods: "))
                    fvc = pmTc1*(((1+rc1)^(tc1)-1)/rc1) + (qmTc/rc1)*((((1+rc1)^(tc1)-1)/rc1)-tc1)
                    
                    
                    print(paste("YOUR FUTURE VALUE IS, FV = ", fvc))
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                  }
              
              }else if (SSK1==2){
                menu(c("PRESENT VALUE", "FUTURE VALUE"), title=" WHAT DO YOU WANT TO CALCULATE? Choose a value for S1:")
                SK1 <- as.numeric (readline(prompt="Enter the value of S1: "))
                
                if (SK1==1){
                  print('YOU WANT TO CALCULATE THE PRESENT VALUE OF YOUR NONLEVEL ANNUITY WITH DECREASING ARITHMETIC PROGRESSION.')
                  rx <-  as.numeric(readline(prompt="Enter the Effective Interest Rate per Interest Period: "))
                  tx <-  as.integer(readline(prompt="Enter the Number of Interest Periods: "))
                  pvx = (tx - ((1-(1+rx)^(-tx))/rx))/rx
                  print(paste("YOUR PRESENT VALUE IS, PV = ", pvx))
                  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                } else if (SK1==2){
                  print('YOU WANT TO CALCULATE THE FUTERE VALUE OF YOUR NONLEVEL ANNUITY WITH DECREASING ARITHMETIC PROGRESSION.')
                  rx1 <-  as.numeric(readline(prompt="Enter the Effective Interest Rate per Interest Period: "))
                  tx1 <-  as.integer(readline(prompt="Enter the Number of Interest Periods: "))
                  fvx = ((1+rx1)^(tx1))*((tx1 - ((1-(1+rx1)^(-tx1))/rx1))/rx)
                  
                  print(paste("YOUR FUTURE VALUE IS, FV = ", fvx))
                  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                }
                
              }
            }   
  }
    
 
 
    
  
} else if (D1==2){
  print('Welcome to the Annuity DUE Module.')
  
  menu(c("LEVEL ANNUITIES", "NONLEVEL ANNUITIES"), title="BELOW ARE TWO OPTIONS YOU SHOULD CHOOSE TO PROCEED WITH YOUR COMPUTATION. FIRST ENTER 0 FROM YOUR KEYBOARD FOR THE SELECTION OPTION THAT FOLLOWS THIS LIST, THEN RUN THE PROGRAM ONE TIME. SECONDLY, ENTER THE NUMBER THAT CORRESPONDS TO THE OPTION OF YOUR CHOICE FROM YOUR KEYBOARD AND THEN RUN YOUR PROGRAM AGAIN:" )
  print('!!!!!!!!!!!!WARNING!!!!!!!!!! MAKE SURE YOUR CHOICE IS EITHER 1 OR 2 !!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!')
  Z <- as.numeric (readline(prompt="ENTER THE NUMBER FOR SELECTED CHOICE? "))
  
      if (Z==1){
        menu(c("PRESENT VALUE", "FUTURE VALUE", "PAYMENT", "INVESTMENT TIME", "RATE"), title=" WHAT DO YOU WANT TO CALCULATE? Choose a value for S1:")
        SZ1 <- as.numeric (readline(prompt="Enter the value of S1: "))
        
        if (SZ1==1){
          print('YOU WANT TO CALCULATE THE PRESENT VALUE OF YOUR ANNUITY DUE WITH LEVEL PAYMENTS.')
          Pmt <-  as.integer(readline(prompt="Enter the Payment Amount/Contribution: "))
          R <-  as.numeric(readline(prompt="Enter the Rate per Interest Period: "))
          d <-  as.numeric(readline(prompt="Enter the Discount Rate per Interest Period: "))
          T <-  as.integer(readline(prompt="Enter the Investment time: "))
          Pv = Pmt*((1-(1+R)^(-T))/d)
          print(paste("YOUR PRESENT VALUE IS, PV = ", Pv))
          print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        } else if (SZ1==2){
          print('YOU WANT TO CALCULATE THE FUTERE VALUE OF YOUR ANNUITY DUE WITH LEVEL PAYMENTS.')
          Pmt1 <-  as.integer(readline(prompt="Enter the Payment Amount/Contribution: "))
          R1 <-  as.numeric(readline(prompt="Enter the Rate per Interest Period: "))
          d1 <-  as.numeric(readline(prompt="Enter the Discount Rate per Interest Period: "))
          T1 <-  as.integer(readline(prompt="Enter the Investment time: "))
          Fv = Pmt1*(((1+R1)^(T1)-1)/d1)
          print(paste("YOUR FUTURE VALUE IS, FV = ", Fv))
          print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        } else if (SZ1==3){
          menu(c("IF YOU WANT TO USE PRESENT VALUE", "IF YOU WANT TO USE FUTURE VALUE"), title="Choose a value for S1 to proceed with the calculation of Payment/Contribution, t:")
          SSZ2 <- as.numeric (readline(prompt="Enter the value of S1: "))
          
            if (SSZ2==1){
              print('YOU WANT TO CALCULATE THE Payment/Contribution(LEVEL PAYMENT) OF YOUR ANNUITY DUE USING THE PRESENT VALUE.')
              R2 <-  as.numeric(readline(prompt="Enter the Rate per Interest Period: "))
              T2 <-  as.integer(readline(prompt="Enter the Investment time: "))
              d2 <-  as.numeric(readline(prompt="Enter the Discount Rate per Interest Period: "))
              Pv2 = (1-(1+R2)^(-T2))/d2
              Pv3 <-  as.integer(readline(prompt="Enter the Amount you wish to get(borrow/PRESENT VALUE): "))
              Ppmt = Pv3/Pv2
              print(paste("YOUR PAYMENT/CONTRIBUTION IS, PMT = ", Ppmt))
              print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
              
            } else if (SSZ2==2){
              print('YOU WANT TO CALCULATE THE Payment/Contribution(LEVEL PAYMENTS) OF YOUR ANNUITY DUE USING THE FUTURE VALUE.')
              R3 <-  as.numeric(readline(prompt="Enter the Rate per Interest Period: "))
              T3 <-  as.integer(readline(prompt="Enter the Investment time: "))
              d3 <-  as.numeric(readline(prompt="Enter the Discount Rate per Interest Period: "))
              Fv1 = ((1+R3)^(T3)-1)/d3
              Pv4 <-  as.integer(readline(prompt="Enter the Amount you wish to Accoumulate/FUTURE VALUE: "))
              Ppmt1 = Pv4/Fv1
              print(paste("YOUR PAYMENT/CONTRIBUTION IS, PMT = ", Ppmt1))
              print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            }
            
        } else if (SZ1==4){
          menu(c("IF YOU WANT TO USE PRESENT VALUE", "IF YOU WANT TO USE FUTURE VALUE"), title="Choose a value for S1 to proceed with the calculation of your investment time, t:")
          SZ2 <- as.numeric (readline(prompt="What is the value of S1? "))
          
            if (SZ2==1){
              print('YOU WANT TO CALCULATE THE INVESTMENT TIME OF YOUR ANNUITY DUE MADE OF LEVEL PAYMENTS USING THE PRESENT VALUE.')
              Pv5 <-  as.integer(readline(prompt="Enter Present value: "))
              Pmt5 <-  as.integer(readline(prompt="Enter the Payment/Conribution: "))
              R5 <-  as.numeric(readline(prompt="Enter the rate: "))
              d5 <-  as.numeric(readline(prompt="Enter the Discount Rate per Interest Period: "))
              T5 = -1*(log(1-(d5*Pv5/Pmt5))/log(1+R5))
              print(paste("YOUR INVESTMENT TIME IS, T = ", T5))
              print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
              
            } else if (S2==2){
              print('YOU WANT TO CALCULATE THE INVESTMENT TIME OF YOUR ANNUITY DUE MADE OF LEVEL PAYMENTS USING THE FUTURE VALUE.')
              Fv6 <-  as.integer(readline(prompt="Enter future value: "))
              Pmt6 <-  as.integer(readline(prompt="Enter the Payment/Conribution: "))
              R6 <-  as.numeric(readline(prompt="Enter the rate: "))
              d6 <-  as.numeric(readline(prompt="Enter the Discount Rate per Interest Period: "))
              T6 = log(1+(d6*Fv6/Pmt6))/log(1+R6)
              print(paste("YOUR INVESTMENT TIME IS, T = ", T6))
              print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            }
            
          } else if (S1==5){
            print('YOU WANT TO CALCULATE THE INVESTMENT EFFECTIVE INTEREST RATE OF YOUR ANNUITY DUE MADE OF LEVEL PAYMENTS.')
            print('SORRY THIS MODULE IS NOT CURRENTLY AVAILABLE.')
            
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
          }
        
        
      } else if (Z==2){
        
        menu(c("FOR INCREASING ARITHMETIC PROGRESSION", "FOR DECREASING ARITHMETIC PROGRESSION"), title=" WHAT DO YOU WANT TO CALCULATE? Choose a value for S1:")
        SSKZ1 <- as.numeric (readline(prompt="Enter the value of S1: "))
        if (SSKZ1==1){
          
          menu(c("PRESENT VALUE", "FUTURE VALUE"), title=" WHAT DO YOU WANT TO CALCULATE? Choose a value for S1:")
          SKZ1 <- as.numeric (readline(prompt="Enter the value of S1: "))
          
          if (SKZ1==1){
            print('YOU WANT TO CALCULATE THE PRESENT VALUE OF YOUR NONLEVEL ANNUITY WITH INCREASING ARITHMETIC PROGRESSION.')
            PmTZC <-  as.integer(readline(prompt="Enter the Payment Amount/Contribution: "))
            qmTZc <-  as.integer(readline(prompt="Enter the  Amount/Contribution of increase: "))
            rZc <-  as.numeric(readline(prompt="Enter the Effective Interest Rate per Interest Period: "))
            dZ <-  as.numeric(readline(prompt="Enter the Discount Rate per Interest Period: "))
            tZc <-  as.integer(readline(prompt="Enter the Number of Interest Periods: "))
            PvZ = PmTZC*((1-(1+rZc)^(-tZc))/dZ) + (qmTZc/dZ)*(((1-(1+rZc)^(-tZc))/rZc)-(tZc)*(1+rZc)^(-tZc))
            print(paste("YOUR PRESENT VALUE IS, PV = ", PvZ ))
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
          } else if (SKZ1==2){
            print('YOU WANT TO CALCULATE THE FUTERE VALUE OF YOUR NONLEVEL ANNUITY WITH INCREASING ARITHMETIC PROGRESSION.')
            pmTZc1 <-  as.integer(readline(prompt="Enter the Payment Amount/Contribution: "))
            qmTZc1 <-  as.integer(readline(prompt="Enter the  Amount/Contribution of increase: "))
            rZc1 <-  as.numeric(readline(prompt="Enter the Rate per Interest Period: "))
            dZ1 <-  as.numeric(readline(prompt="Enter the Discount Rate per Interest Period: "))
            tZc1 <-  as.integer(readline(prompt="Enter the Number of Interest Periods: "))
            fvcZ = pmTZc1*(((1+rZc1)^(tZc1)-1)/dZ1) + (qmTZc1/dZ1)*((((1+rZc1)^(tZc1)-1)/dZ1)-tZc1)
            print(paste("YOUR FUTURE VALUE IS, FV = ", fvcZ))
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
          }
          
        }else if (SSKZ1==2){
          menu(c("PRESENT VALUE", "FUTURE VALUE"), title=" WHAT DO YOU WANT TO CALCULATE? Choose a value for S1:")
          SKZ1 <- as.numeric (readline(prompt="Enter the value of S1: "))
          
            if (SKZ1==1){
              print('YOU WANT TO CALCULATE THE PRESENT VALUE OF YOUR NONLEVEL ANNUITY WITH DECREASING ARITHMETIC PROGRESSION.')
              rZx <-  as.numeric(readline(prompt="Enter the Effective Interest Rate per Interest Period: "))
              tZx <-  as.integer(readline(prompt="Enter the Number of Interest Periods: "))
              dZx <-  as.numeric(readline(prompt="Enter the Discount Rate per Interest Period: "))
              pvZx = (tZx - ((1-(1+rZx)^(-tZx))/rZx))/dZx
              print(paste("YOUR PRESENT VALUE IS, PV = ", pvZx))
              print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            } else if (SKZ1==2){
              print('YOU WANT TO CALCULATE THE FUTERE VALUE OF YOUR NONLEVEL ANNUITY WITH DECREASING ARITHMETIC PROGRESSION.')
              rZx1 <-  as.numeric(readline(prompt="Enter the Effective Interest Rate per Interest Period: "))
              tZx1 <-  as.integer(readline(prompt="Enter the Number of Interest Periods: "))
              dZx1 <-  as.numeric(readline(prompt="Enter the Discount Rate per Interest Period: "))
              fvZx = ((1+rZx1)^(tZx1))*((tZx1 - ((1-(1+rZx1)^(-tZx1))/rZx1))/dZx1)
              print(paste("YOUR FUTURE VALUE IS, FV = ", fvZx))
              print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            }
          
        }
      }   
  
  
} else{
  print('!!!!!!!!!!!!!!!WARNING!!!!!!!!! MAKE SURE YOUR CHOICE IS EITHER 1 OR 2 !!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!')
  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  
}

  i = i+1
}










