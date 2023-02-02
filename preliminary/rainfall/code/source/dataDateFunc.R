
#date defining function
dataDateMkr <- function(dateVar=NA){
  #try get date from source if exist 
  globalDate<-commandArgs(trailingOnly=TRUE)[1] #pull from outside var when sourcing script
  #make globalDate if 
  globalDate<-if(is.na(globalDate) & !is.na("dateVar")){
            as.Date(dateVar) #if globalDate is NA & dateVar is NA, set date in code
            }else{
            ifelse(exists("globalDate"),globalDate,NA) #else define globalDate as NA if it does exist
            } 
  dataDate<-as.Date(ifelse(!is.na(globalDate),globalDate-1,Sys.Date()-1)) #make dataDate from globalDate or sysDate -1
  return(dataDate)
  }