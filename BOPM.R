
i <- 1
while (i < 10) {
  
  print('Welcome to an R Program for Calls and Puts problem in Finance.')
 
  #c1 <-  as.numeric(readline(prompt="Enter the number of stock prices you wish to consider: "))
  INPT1 = numeric()
  X1 <-  as.numeric(readline(prompt=": ENTER THE MEAN RETURN PER YEAR:"))
  INPT1 <- append(INPT1,X1)
  X2 <-  as.numeric(readline(prompt=": ENTER THE STANDARD DEVIATION OF ANNUAL RETURN:"))
  INPT1 <- append(INPT1,X2)
  X3 <-  as.numeric(readline(prompt=": ENTER THE ANNUAL INTEREST RATE:"))
  INPT1 <- append(INPT1,X3)
  X4 <-  as.numeric(readline(prompt=": ENTER THE INITIAL STOCK PRICE:"))
  INPT1 <- append(INPT1,X4)
  X5 <-  as.numeric(readline(prompt=": ENTER THE OPTION EXERCISE PRICE:"))
  INPT1 <- append(INPT1,X5)
  X6 <-  as.numeric(readline(prompt=": ENTER THE OPTION EXERCISE DATE:"))
  INPT1 <- append(INPT1,X6)
  X7 <-  as.numeric(readline(prompt=": ENTER THE NUMBER OF DIVISIONS PER YEAR:"))
  INPT1 <- append(INPT1,X7)
  print(paste("ALL YOUR INPUT VARIABLES ARE:"))
  print(INPT1)
  
  X8 <- (1/X7)
  print(paste("The length of one division is:", X8))
  X9 <- exp(X1*X8+X2*sqrt(X8))
  print(paste("THE UP MOVE PER LENGHT OF ONE DIVISION:", X9))
  X10 <- exp(X1*X8-X2*sqrt(X8))
  print(paste("THE DOWN MOVE PER LENGHT OF ONE DIVISION:", X10))
  X11 <- exp(X3*X8)
  print(paste("THE INTEREST RATE PER LENGHT OF ONE DIVISION IS:", X11))
  X12 <- round(X6*X7)
  print(paste("THE PERIODS UNTIL MATURITY IS, n = :", X12))
  X13 = 1+X3
  print(paste("THE INTEREST RATE, R = :", X13))
  spu = round((X13 - X10)/(X13*(X9-X10)),3)
  print(paste("THE UPWARD STATE PRICE IS = :", spu))
  spd = round((X9 - X13)/(X13*(X9-X10)),3)
  print(paste("THE UPWARD STATE PRICE IS = :", spd))
  
 
  
  menu(c("AMERICAN CALL OPTION", "EUROPEAN PUT OPTION", "EUROPEAN CALL OPTION", "AMERICAN PUT OPTION"), title="BELOW ARE FOUR OPTIONS YOU SHOULD CHOOSE TO PROCEED WITH YOUR COMPUTATION. 
       FIRST ENTER 0 FROM YOUR KEYBOARD FOR THE SELECTION OPTION THAT FOLLOWS THIS LIST, THEN RUN THE PROGRAM ONE TIME. SECONDLY, ENTER THE NUMBER THAT CORRESPONDS TO THE OPTION OF YOUR CHOICE FROM YOUR KEYBOARD AND THEN RUN YOUR PROGRAM AGAIN:" )
  print('!!!!!!!!!!!!WARNING!!!!!!!!!! MAKE SURE YOUR CHOICE IS EITHER 1, 2, 3 OR 4 !!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!')
  D1 <- as.numeric (readline(prompt="ENTER THE NUMBER OF THE SELECTED CHOICE? "))
  
  if (D1==1){
    print('WELCOME TO THE AMERICAN CALL OPTION MODULE.')
    print('YOU WANT TO CALCULATE THE FAIR PRICE OF THIS OPTION.')
    ph = put_price_am_bin <- am_call_bin(X4, X5, X3, X2, X6, X7)
    Ph1 = round(ph,2)
    print(paste("THE AMERICAN CALL OPTION PRICE IS = :", Ph1))
  
} else if (D1==2){
    print('WELCOME TO THE EUROPEAN PUT OPTION MODULE.')
    print('YOU WANT TO CALCULATE THE FAIR PRICE OF THIS OPTION.')
    d <- 0
    PU1 = round(bsput(X4, X5, X2, X3, X6,d),3)
    print(paste("THE EUROPEAN PUT OPTION PRICE IS = :", PU1))
    
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
} else if (D1==3){
  print('WELCOME TO THE EUROPEAN CALL OPTION MODULE.')
  print('YOU WANT TO CALCULATE THE FAIR PRICE OF THIS OPTION.')
  p = call_price_am_bin <- am_call_bin(X4, X5, X3, X2, X6, X7)
  P1 = round(p,2)
  print(paste("THE EUROPEAN CALL OPTION PRICE IS = :", P1))
  
  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  
} else if (D1==4){
  print('WELCOME TO THE AMERICAN PUT OPTION MODULE.')
  print('YOU WANT TO CALCULATE THE FAIR PRICE OF THIS OPTION.')
  d <- 0
  PU = round(bscall(X4, X5, X2, X3, X6,d),3)
  print(paste("THE AMERICAN PUT OPTION PRICE IS = :", PU))
  
  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  
}
  
    
} 

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  


















