#library(shiny)
#Author: Stevye Tinchie F.
#Date: 10/4/2018

i <- 1
while (i < 3) {
  
  print('Welcome to an R Program for Calls and Puts problem in Finance.')
  
  #c1 <-  as.numeric(readline(prompt="Enter the number of stock prices you wish to consider: "))
  #INPT1 = numeric()
  #X1 <-  as.numeric(readline(prompt=": ENTER THE INITIAL STOCK PRICE:"))
  #INPT1 <- append(INPT1,X1)
  
  XH <-  as.numeric(readline(prompt=": ENTER THE NUMBER OF STOCK PRICES YOU WANT TO CONSIDER:"))
  INPT1G = numeric()
  INPT2G = numeric()
  INPT3G = numeric()
  INPT4G = numeric()
  INPT5G = numeric()
  d1 = numeric()
  d2 = numeric()
  calls = numeric()
  puts = numeric()
  cbang = numeric()
  pbang = numeric()
  i <- 1
  while (i < XH+1) {
    X1 <-  as.numeric(readline(prompt=": ENTER THE INITIAL STOCK PRICE:"))
    INPT1G = append(INPT1G,X1)
    i = i+1
  }
  print(paste("ALL YOUR STOCK PRICES ARE:"))
  print(INPT1G)
  
  i <- 1
  while (i < XH+1) {
    X2 <-  as.numeric(readline(prompt=": ENTER THE THE OPTION EXERCISE PRICE:"))
    INPT2G = append(INPT2G,X2)
    i = i+1
  }
  print(paste("ALL YOUR EXERCISE PRICES ARE:"))
  print(INPT2G)
  
  i <- 1
  while (i < XH+1) {
    X3 <-  as.numeric(readline(prompt=": ENTER THE INTEREST RATE:"))
    INPT3G <- append(INPT3G,X3)
    i = i+1
  }
  print(paste("ALL YOUR INTEREST RATES ARE:"))
  print(INPT3G)
  
  i <- 1
  while (i < XH+1) {
    X4 <-  as.numeric(readline(prompt=": ENTER THE TIME TO MATURITY:"))
    INPT4G <- append(INPT4G,X4)
    i = i+1
  }
  print(paste("ALL YOUR TIME TO MATURITY ARE:"))
  print(INPT4G)
  
  
  i <- 1
  while (i < XH+1) {
    X5 <-  as.numeric(readline(prompt=": ENTER SIGMA:"))
    INPT5G <- append(INPT5G,X5)
    i = i+1
  }
  print(paste("ALL YOUR VOLATILITIES ARE:"))
  print(INPT5G)
  
  j <- 1
  while (j < XH+1) {
    
    
    dONE <- (log(INPT1G[j]/INPT2G[j])+INPT3G[j]*INPT4G[j])/(INPT5G[j]*sqrt(INPT4G[j]))+0.5*INPT5G[j]*sqrt(INPT4G[j])
    d1 <- append(d1,dONE)
    j=j+1
  }
  print(paste("ALL YOUR D1 ARE:"))
  print(d1)
  
  j <- 1
  while (j < XH+1) {
    
    dTWO = d1[j]-INPT5G[j]*sqrt(INPT4G[j])
    d2 <- append(d2,dTWO)
    j=j+1
  }
  print(paste("ALL YOUR D2 ARE:"))
  print(d2)
  
  j <- 1
  while (j < XH+1) {
    
    BSCALL = INPT1G[j]*pnorm(d1[j]) - INPT2G[j]*exp(-INPT4G[j]*INPT3G[j])*pnorm(d2[j])
    calls <- append(calls,BSCALL)
    j=j+1
  }
  print(paste("ALL YOUR BSCALLS ARE:"))
  print(calls)
    
  j <- 1
  while (j < XH+1) {
    
    BSPUT = INPT2G[j]*exp(-INPT4G[j]*INPT3G[j])*pnorm(-d2[j]) - INPT1G[j]*pnorm(-d1[j])
    puts <- append(puts,BSPUT)
    j=j+1
  }
  print(paste("ALL YOUR BSPUT ARE:"))
  print(puts)
  
  j <- 1
  while (j < XH+1) {
    CALLBANG = pnorm(d1[j])*INPT1G[j]/calls[j]
    cbang = append(cbang,CALLBANG)
    j=j+1
  }
  print(paste("ALL YOUR CALLBANGS ARE:"))
  print(cbang)
  
  j <- 1
  while (j < XH+1) {
    
    PUTBANG = pnorm(-d1[j])*INPT2G[j]/puts[j]
    pbang = append(pbang,PUTBANG)
    j = j+1
  }
  print(paste("ALL YOUR PUTBANGS ARE:"))
  print(pbang)
  plot(cbang,INPT2G,main = "PAYOFF OF YOUR LONG CALL OPTION",type="l",col="red")
  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PROGRAM TERMINATED SUCCESSFULLY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  
}
















