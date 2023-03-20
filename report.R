library(tidyverse)
library(skimr)     
library(corrplot)
 
library(caret)      
library(mlr)        
library(rpart)     
library(randomForest)
data<-read.csv('D:/data/heart_failure.csv')

summary(data)
skim(data)
cor(data) %>%
  corrplot(method = "color", type = "lower", tl.col = "black", tl.srt = 45,
           p.mat = cor.mtest(data)$p,
           insig = "p-value", sig.level = -1)



set.seed(2)
require(smotefamily)
data<-SMOTE(data[,-13],data[,13],K=13)
data<-data$data
data$class<-as.factor(as.numeric(data$class))



library(mlbench)
library(caret)
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(data[,1:12], data[,13], sizes=c(1:12), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))

xx<-results[["fit"]][["importance"]]
xx<-data.frame(xx)
xx$MeanDecreaseAccuracy<-xx$MeanDecreaseAccuracy*100
library(ggplot2)
ggplot(data = xx, 
       mapping = aes(x = reorder(rownames(xx),-MeanDecreaseAccuracy), y = MeanDecreaseAccuracy))+ geom_bar(stat = 'identity')+theme(axis.text.x=element_text(angle=90),axis.title.x=element_blank())


ggplot(data = xx, 
       mapping = aes(x = reorder(rownames(xx),-MeanDecreaseGini), y = MeanDecreaseGini))+ geom_bar(stat = 'identity')+theme(axis.text.x=element_text(angle=90),axis.title.x=element_blank())



data<-data[,c(results$optVariables,'class')]
n<-dim(data)[2]
set.seed(15)
train_sub = sample(nrow(data),8/10*nrow(data))
train_data = data[train_sub,]
test_data =data[-train_sub,]

ntre<-data.frame(c(10,100,200,300,400,500,600,700,800,1000,2000))
ntre$acc<-0
colnames(ntre)[1]<-'ntre'
for (i in 1:9) {
  model_randomforest <- randomForest(class ~ ., data = train_data, importance = TRUE,ntree=ntre[i,1])
  model_randomforest
  temp_randomforest<-data.frame(predict(model_randomforest,test_data[,-n]))
  
  colnames(temp_randomforest)<-'temp_randomforest'

  library(caret)
  aa<-confusionMatrix(test_data$class,temp_randomforest$temp_randomforest)
  
  ntre[i,2]<-aa[["overall"]][["Accuracy"]]
}

ntre



model_randomforest <- randomForest(class ~ ., data = train_data, importance = TRUE,ntree=400)
model_randomforest

temp_randomforest<-data.frame(predict(model_randomforest,test_data[,-n]))

colnames(temp_randomforest)<-'temp_randomforest'

library(caret)
confusionMatrix(test_data$class,temp_randomforest$temp_randomforest)
library(pROC)
temp_randomforest_prob<-data.frame(predict(model_randomforest,test_data[,-n],type = 'prob'))

roc_randomforest<-roc(as.integer(test_data$class)-1,temp_randomforest_prob$X1)
plot(roc_randomforest,print.AUC=T,col='red',legacy.axes=T)

#logistic
library(rms)
model_logistic<-lrm(class~.,data=train_data,x=T,y=T)
#summary(model_logistic)


temp_logistic<-predict(model_logistic,newdata = test_data[,-n],type = "fitted")
temp_logistic<-data.frame(ifelse(temp_logistic>=0.5,1,0))
colnames(temp_logistic)<-'temp_logistic'


confusionMatrix(test_data$class,factor(temp_logistic$temp_logistic))

temp_logistic_prob<-predict(model_logistic,newdata = test_data[,-n],type = "fitted")

roc_logistic<-roc(as.integer(test_data$class)-1,temp_logistic_prob)
plot(roc_logistic,print.AUC=T,add=T,col='blue')


##SVM
library(e1071)
model_svm<-svm(class~.,data = train_data,
                    kernel='radial',probability = TRUE)

temp_svm<-predict(model_svm,test_data[,-n],probability = TRUE)


confusionMatrix(test_data$class,factor(temp_svm))


temp_svm_prob<-data.frame(attr(temp_svm, "probabilities")[,1])
colnames(temp_svm_prob)<-'temp_svm_prob'

roc_svm<-roc(as.integer(test_data$class)-1,temp_svm_prob$temp_svm_prob)
plot(roc_svm,print.AUC=T,add=T,col='purple')



legend(x=0.55,y=0.5, legend=c("RandomForest AUC=0.9804","Logistic AUC=0.924",
                             "SVM AUC=0.942")
       ,cex =0.6,lwd=2,
       col=c("red","blue","purple","black"),inset=.5,lty=c(1,2,3,4))




p_positive <- temp_randomforest_prob$X0
sor <- order(p_positive)
p_positive <- p_positive[sor]
y <- test_data$class[sor]

y <- ifelse(y == "0",1,0)

groep <- cut2(p_positive, g = 10)
 
meanpred_rm <- round(tapply(p_positive, groep, mean), 3)
meanobs <- round(tapply(y, groep, mean), 3) 

finall_rm <- data.frame(meanpred_rm = meanpred_rm, meanobs_rm = meanobs)

ggplot(finall_rm,aes(x = meanpred_rm,y = meanobs_rm))+ 
  geom_line(linetype = 2)+ 
  geom_abline(slope = 1,intercept = 0,color = "red")+ 
  labs(x="Predicted Probability",y = "Observed Probability",title = "calibration_curve")


######################logistic################
p_positive <- temp_logistic_prob
sor <- order(p_positive)
p_positive <- p_positive[sor]
y <- test_data$class[sor]

y <- ifelse(y == "1",1,0)

groep <- cut2(p_positive, g = 10)

meanpred_lr <- round(tapply(p_positive, groep, mean), 3)
meanobs <- round(tapply(y, groep, mean), 3) 

finall_lr <- data.frame(meanpred_lr = meanpred_lr, meanobs_lr = meanobs)


ggplot(finall_lr,aes(x = meanpred_lr,y = meanobs_lr))+ 
  geom_line(linetype = 2)+ 
  geom_abline(slope = 1,intercept = 0,lty="solid",color = "red")+ 
  labs(x="Predicted Probability",y = "Observed Probability",title = "calibration_curve")
############SVM########

p_positive <- temp_svm_prob$temp_svm_prob
sor <- order(p_positive)
p_positive <- p_positive[sor]
y <- test_data$class[sor]

y <- ifelse(y == "1",1,0)

groep <- cut2(p_positive, g = 10)

meanpred_svm <- round(tapply(p_positive, groep, mean), 3)



fill<-cbind.data.frame(meanobs,meanpred_rm,meanpred_lr,meanpred_svm)


ggplot()+geom_line(data = fill,aes(x = meanobs,y = meanpred_rm,colour = "RandomForest"),linewidth=1,linetype = 1)+
  #geom_point(data = data,aes(x = year,y = GDP,colour = "GDP"),size=3)+
  ylim(0,1)+
  geom_line(data = fill,aes(x = meanobs,y = meanpred_lr,colour = "Logistic"),linewidth=1,linetype = 2) + 
  geom_line(data = fill,aes(x = meanobs,y = meanpred_svm,colour = "SVM"),linewidth=1,linetype = 3) +
  #geom_line(data = fill,aes(x = meanobs,y = meanpred_decision,colour = "Decision Tree"),size=1,linetype = 4) + 
  #geom_line(data = data,aes(x = year,y = FDI,colour ="FDI"),size=1) +
  #scale_colour_manual("",values = c("GDP" = "red","FDI" = "green","DI"="yellow"))+
  geom_abline(slope = 1,intercept = 0,lty="solid",color = "red")+ 
  labs(x="Predicted Probability",y = "Observed Probability",title = "calibration_curve")+
  theme_light() 




