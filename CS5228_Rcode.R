# import package
library(ranger)
library(MICE)
library(ROSE)
library(ggplot2)

# create functions for data cleaning & exploration
freq.fun = function(name, data)
{
  cat.var = data[, name]
  k <- as.data.frame(table(cat.var, training$y),nrow=length(levels(cat.var)))
  names(k)[2] <- 'Response'
  k <- ddply(k, .(cat.var), transform, percent = Freq/sum(Freq) * 100)
  k <- ddply(k, .(cat.var), transform, pos = (cumsum(Freq) - 0.5 * Freq))
  k$label <- paste0(sprintf("%.0f", k$percent), "%")
  k$label <- ifelse(k$Freq<25 | k$percent < 50, "", k$label) #hard-coded values, change as required
  return(k)
}


freq.prop = function(name, data)
{
  mytable = table(data[, name], data$y)
  myproptable = prop.table(mytable, 1)
  return(data.frame(val = as.integer(row.names(myproptable)), 
                    total = mytable[, "no"] + mytable[,"yes"],
                    no = mytable[,"no"],
                    yes = mytable[, "yes"],
                    success_rate = myproptable[,"yes"]*100))
}

summary_of_unknown = function(data){
  names = colnames(data)
  number_of_na = rep(0,length(names))
  for(i in seq(0, length(names), 1)){
    print(i)
    if(!is.na(summary(data[,names[i]])["unknown"]))
      number_of_na[i] = summary(data[,names[i]])["unknown"]
  }
  return(data.frame(names, number_of_na))
}

drop_variable = function(var, data){
  data_drop = data[ , !(names(data) %in% var)]
  return(data_drop)
}

DStoBN = function(data, DSval, isYes){
  list = data == DSval
  if(isYes)
  {
    data[list] = "yes"
    data[!list] = "no"
  }
  else{
    data[!list] = "yes"
    data[list] = "no"
  }
  data = as.factor(data)
  return(data)
}

unknowtoNA = function(data){
  list = data == "unknown"
  data[list] = NA
  return(data)
}

# data exploration
age_total = hist(training$age, xlim=c(0,100),col='skyblue',border=F,main = "Histogram of age", xlab = "age")
age_yes = hist(training$age[training$y =="yes"],add=T,col=scales::alpha('red',.5),border=F)
legend("topright", c("No", "Yes"), fill=c("skyblue", scales::alpha('red',.5)))

duration_total = hist(training$duration, xlim=c(0,1500), breaks = 50,
                      col='skyblue',border=F,main = "Histogram of duration", xlab = "last contact duration")
duration_yes = hist(training$duration[training$y =="yes"],breaks = 50,
                    add=T,col=scales::alpha('red',.5),border=F)
legend("topright", c("No", "Yes"), fill=c("skyblue", scales::alpha('red',.5)))


p_job = ggplot(freq.fun("job", training), aes(x = cat.var, y = Freq, fill = Response)) + 
  geom_bar(stat = "identity", width = 0.7) + xlab(NULL) + ylab(NULL) + coord_flip() +
  ggtitle('Job') + theme(plot.title=element_text(size=15))+
  geom_text(aes(y = pos, label = label), size=2)

p_job
# to handle the missing value in job

p_marital = ggplot(freq.fun("marital", training), aes(x = cat.var, y = Freq, fill = Response)) + 
  geom_bar(stat = "identity", width = 0.7) + xlab(NULL) + ylab(NULL) + coord_flip() +
  ggtitle('Marital') + theme(plot.title=element_text(size=15))+
  geom_text(aes(y = pos, label = label), size=3)
p_marital
# ignore marital status

p_education = ggplot(freq.fun("education", training), aes(x = cat.var, y = Freq, fill = Response)) + 
  geom_bar(stat = "identity", width = 0.7) + xlab(NULL) + ylab(NULL) + coord_flip() +
  ggtitle('Education') + theme(plot.title=element_text(size=15))+
  geom_text(aes(y = pos, label = label), size=3)
p_education

mytable_education = table(training$education, training$y)
mytable_education
# to handle the missing values in education

p_default = ggplot(freq.fun("default", training), aes(x = cat.var, y = Freq, fill = Response)) + 
  geom_bar(stat = "identity", width = 0.7) + xlab(NULL) + ylab(NULL) + coord_flip() +
  ggtitle('Default') + theme(plot.title=element_text(size=15))+
  geom_text(aes(y = pos, label = label), size=3)
p_default
mytable_default = table(training$default, training$y)
chisq.test(mytable_default[1:2,])

# to do chi-square test to show if there is a strong evidence between the UNKNOW status and the response value
# if yes, keep UNKNOWN as a category

p_housing = ggplot(freq.fun("housing", training), aes(x = cat.var, y = Freq, fill = Response)) + 
  geom_bar(stat = "identity", width = 0.7) + xlab(NULL) + ylab(NULL) + coord_flip() +
  ggtitle('Housing') + theme(plot.title=element_text(size=15))+
  geom_text(aes(y = pos, label = label), size=3)
p_housing
# ignore the condition if the person his/her own house or not

p_loan = ggplot(freq.fun("loan", training), aes(x = cat.var, y = Freq, fill = Response)) + 
  geom_bar(stat = "identity", width = 0.7) + xlab(NULL) + ylab(NULL) + coord_flip() +
  ggtitle('Loan') + theme(plot.title=element_text(size=15))+
  geom_text(aes(y = pos, label = label), size=3)
p_loan
# ignore the condition if the person has taken up a loan before or not

p_contact = ggplot(freq.fun("contact", training), aes(x = cat.var, y = Freq, fill = Response)) + 
  geom_bar(stat = "identity", width = 0.7) + xlab(NULL) + ylab(NULL) + coord_flip() +
  ggtitle('Contact') + theme(plot.title=element_text(size=15))+
  geom_text(aes(y = pos, label = label), size=3)
p_contact
# do not ignore and try to find the explanation

p_month = ggplot(freq.fun("month", training), aes(x = cat.var, y = Freq, fill = Response)) + 
  geom_bar(stat = "identity", width = 0.7) + xlab(NULL) + ylab(NULL) + coord_flip() +
  ggtitle('Month') + theme(plot.title=element_text(size=15))+
  geom_text(aes(y = pos, label = label), size=3)
p_month
# may has the highest rate and sep to dec has the lowest

p_day_of_week = ggplot(freq.fun("day_of_week", training), aes(x = cat.var, y = Freq, fill = Response)) + 
  geom_bar(stat = "identity", width = 0.7) + xlab(NULL) + ylab(NULL) + coord_flip() +
  ggtitle('Day of week') + theme(plot.title=element_text(size=15))+
  geom_text(aes(y = pos, label = label), size=3)
p_day_of_week
# ignore this

campaign_total = hist(training$campaign, breaks = 100, ylim = c(0, 50), xlab = "number of campaign",
                      col='skyblue',border=F,main = "Number of contact in the campaign")
campaign_yes = hist(training$campaign[training$y =="yes"],breaks = 50, add=T,col=scales::alpha('red',.5),border=F)
legend("topright", c("No", "Yes"), fill=c("skyblue", scales::alpha('red',.5)))

freq.prop("campaign", training)
plot(freq.prop("campaign", training)$val, freq.prop("campaign", training)$success_rate, 
     type = "b", lwd = 1.5, col = "darkblue", pch = 16, cex = 0.8,
     xlim = c(0,60), ylim = c(0, 13),
     xlab = "Number of contacts during the campaign", ylab = "Rate of success (%)",
     main = "Success Rate vs Number of Contacts")
text(freq.prop("campaign", training)$val, freq.prop("campaign", training)$success_rate, 
     freq.prop("campaign", training)$total, 
     lwd = 1, cex=0.6, pos=4, col="black")
legend("topright", pch = 19, c("Total number clients"), col=c("darkblue"))
# an inverse relationship for small value of the number of contacts

freq.prop("pdays", training)
freq.prop("previous", training)
# both are very biased data, consider making it a binary data

table(training$poutcome,training$y)
# keep nonexistent as a category

# use histogram to observe the numberical data
# consider time series

boxplot(emp.var.rate ~ y, data = training, col = hcl(c(0, 240), 50, 70), 
        main = "Employment Variation Rate Per Client Response", lwd = 1.5,
        xlab = "y", ylab = "emp.var.rate")

emp.var.rate_total = hist(training$emp.var.rate, ylim = c(0, 15000), breaks = 30, col='skyblue',border=F,main = "Histogram of emp.var.rate", xlab = "emp.var.rate")
emp.var.rate_yes = hist(training$emp.var.rate[training$y =="yes"], breaks = 30, add=T,col=scales::alpha('red',.5),border=F)
legend("topright", c("No", "Yes"), fill=c("skyblue", scales::alpha('red',.5)))

####################################### 
# data modelling
#######################################
# convert unknow to NA value 
train$education = unknowtoNA(train$education)
train$job = unknowtoNA(train$job)
train$marital = unknowtoNA(train$marital)
train$housing = unknowtoNA(train$housing)
train$loan = unknowtoNA(train$loan)

test$education = unknowtoNA(test$education)
test$job = unknowtoNA(test$job)
test$marital = unknowtoNA(test$marital)
test$housing = unknowtoNA(test$housing)
test$loan = unknowtoNA(test$loan)


# convert 2 default "yes" to "unknown"
train[train$default == "yes", ]$default = "no"
summary_of_unknown(train)
train.len = dim(train)[1]


test[test$default == "yes", ]$default = "no"
summary_of_unknown(test)
test.len = dim(test)[1]

# convert pdays, previous, and campaign to categorical
train = data.frame(train, pdays.fac = rep(NA, train.len), 
                   previous.fac = rep(NA, train.len), campaign.fac = rep(NA, train.len))
str(train)


test = data.frame(test, pdays.fac = rep(NA, test.len), 
                   previous.fac = rep(NA, test.len), campaign.fac = rep(NA, test.len))
str(test)
dim(test)
# convert pdays to categorical
pdays.fac = train$pdays
pdays.fac = DStoBN(pdays.fac, 999, FALSE)
train$pdays.fac = pdays.fac


pdays.fac = test$pdays
pdays.fac = DStoBN(pdays.fac, 999, FALSE)
test$pdays.fac = pdays.fac

# convert previous to categorical
previous.fac = train$previous
previous.fac[train$previous >= 4] = 4
previous.fac = as.factor(previous.fac)
train$previous.fac = previous.fac


previous.fac = test$previous
previous.fac[test$previous >= 4] = 4
previous.fac = as.factor(previous.fac)
test$previous.fac = previous.fac

# convert campaign to categorical
campaign.fac = train$campaign
campaign.fac[train$campaign >= 18] = 18
campaign.fac = as.factor(campaign.fac)
train$campaign.fac = campaign.fac


campaign.fac = test$campaign
campaign.fac[test$campaign >= 18] = 18
campaign.fac = as.factor(campaign.fac)
test$campaign.fac = campaign.fac

# applying MICE for missing values
train.imp = mice(train[, !(names(train) %in% c("previous", "campaign", "pdays"))], m = 5)
train.mice = complete(train.imp)


test.imp = mice(test[, !(names(test) %in% c("previous", "campaign", "pdays"))], m = 5)
test.mice = complete(test.imp)

# apply re-sampling technique
data_balanced_both = ovun.sample(y ~ ., 
                                 data = train[, !(names(train) %in% c("previous", "campaign", "pdays", "nr.employer", "cons.price.idx"))], 
                                 p = 0.4, method = "both")$data
table(data_balanced_both$y)
new.model = ranger(y ~ ., data = data_balanced_both, mtry = 3,
                   write.forest = TRUE, num.trees = 150) 
y_fitted = predict(new.model, 
                   test.mice[, !(names(test.mice) %in% c("previous", "campaign", "nr.employer", "cons.price.idx"))])

compute_dist(y_fitted$predictions, test.new[,"y"])

