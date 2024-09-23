# ------------------------------------------------------------------------------------
# 번  호: 1
# 함수명: normalize_fn
# 입력값: x(정규화시킬 numeric 형태의 데이터)
# 설  명: numeric 형태의 데이터 정규화(min-max)
# ------------------------------------------------------------------------------------

normalize_fn <- function(x){return ((x - min(x)) / (max(x) - min(x)))}


# ------------------------------------------------------------------------------------
# 번  호: 2
# 함수명: normalize_df_fn
# 입력값: data(정규화시킬 dataframe 형태의 데이터), input.min(최소값), input.max(최대값)
# 설  명: dataframe 형태의 데이터 정규화(min-max)
# ------------------------------------------------------------------------------------

normalize_df_fn <- function(data, input.min, input.max){
    input_df <- as.data.frame(data)
    col.names <- names(input.min)
 
    for (i in 1:length(col.names)){
        input_df[, col.names[i]] <- (input_df[, col.names[i]] - input.min[col.names[i]])/(input.max[col.names[i]] - input.min[col.names[i]])  
    }
    return(input_df)
}


# ------------------------------------------------------------------------------------
# 번  호: 3
# 함수명: DNN_relu_train_sel_para_fn
# 입력값: x(독립변수 값), y(종속변수 값), 
#         Xn.t(정규화를 수행한 독립변수 데이터프레임), Yn.t(정규화를 수행한 종속변수 데이터프레임)
# 설  명: 데이터를 정규화한 후, DNN 모델 학습 및 train dataset으로 예측을 통한 성능검증 
# ------------------------------------------------------------------------------------

DNN_relu_train_sel_para_fn <- function(x, y, Xn.t, Yn.t){ 
    lr <- c(0.8, 0.9)                                                    # learning rate(학습률): 0.8, 0.9 (2가지)
    node.hid <- c(round(length(x[1, ]) * 4), round(length(x[1, ]) * 8))  # Num of node in hidden layer(노드 수): 2가지
    epo <- c(1000, 3000)                                                 # Epoch(학습횟수): 1000, 3000 (2가지)(mini batch가 NA일시)      
    mini.bc <- NA                                                        # Mini batch없는게 더 좋은 성능 보임
    gradi.clip <- c(0.8)                                                 # 역전파 가중치 업데이트 작업의 기울기 크기를 제한하는 값: 0.8       
       
    # DNN
    nn.tr <- list()
    iline <- 0  # Numbering
    allcase <- length(lr) * length(node.hid) * length(epo) * length(mini.bc) * length(gradi.clip) * 1  # 2x2x2x1x1 = 8 
    comp.res <- data.frame(matrix(NA, length(y), allcase + 1))  # 3051행 9열 NA값으로 채워진 df
    para.res <- data.frame(matrix(NA, allcase, 7))  # 8행 7열 NA값으로 채워진 df
    comp.res[, 1] <- y  # 1열은 y값 넣기
    
    for(i in 1:length(lr)){  # i: 1, 2
        for(j in 1:length(node.hid)){  # j: 1, 2
            for(k in 1:length(epo)){  # epo: 1, 2
                for(m in 1:length(mini.bc)){  # mini.bc: 1
                    for(l in 1:length(gradi.clip)){  # gradi.clip: 1
                       
                        # 모델 학습(Hidden layer: 2개)
                        iline <- iline + 1  # iline: 1부터 시작
                        nn.tr[[iline]] <- deepnet(Xn.t,  # 독립변수 
                                                  as.matrix(Yn.t),  # 종속변수
                                                  hiddenLayerUnits = c(node.hid[j], node.hid[j]),  # 노드 수: 28, 56  
                                                  activation = c('relu', 'relu'),               # 활성화 함수 
                                                  reluLeak = 0.01,  # reluLeak: ReLU(Rectified Linear Unit) 함수의 변형 함수 중 하나
                                                                    # ReLU 함수는 입력값이 0보다 크면 그대로 출력하고, 0 이하면 0을 출력하는 함수
                                                                    # reluLeak 함수는 입력값이 0 이하일 때, ReLU와는 달리 0이 아닌 작은 값(여기선 0.01로 지정)을 출력
                                                  modelType = 'regress',  # 손실함수(예측값과 실제값의 차이를 수량화함, 이 값을 바탕으로 가중치와 편향 업데이트) 
                                                                          # option: regress(회귀 / 선형단일단위 출력 레이어 생성, 연속형 값 예측), 
                                                                          #         binary(이진분류 / 단일 단위 시그모이드 활성층 생성, 두 개의 클래스로 분류), 
                                                                          #         multiClass(다중 클래스 분류 / softmax 활성화된 출력 class의 수에 해당하는 단위로 레이서 생성, 두 개 이상의 클래스 분류) 
                                                  eta = lr[i],                    # 역전파에 대한 학습률(여기선 0.8, 0.9)
                                                  iterations = epo[k],            # 역전파의 반복 또는 epoch의 수(여기선 1000, 3000) 
                                                  gradientClip = gradi.clip[l],   # 역전파 가중치 업데이트 작업의 기울기 크기를 제한하는 값(여기선 0.8) 
                                                  miniBatchSize = mini.bc[m],  # mini bach 크기(여기선 1) 
                                                  optimiser = 'adam',          # 옵티마이저(option: gradientDescent, momentum, rmsProp, adam)
                                                  normalise = F,               # 정규화 필요 유무 
                                                  stopError = 0.0005)          # 반복을 중지할 수 있는 RMSE값(여기선 0.0005, RMSE값이 0.0005이상이면 중지)
                        
                        Ym.t.nn <- predict.deepnet(nn.tr[[iline]], newData = Xn.t)  # 예측값
                        Ym.t.nn <- Ym.t.nn * (max(y) - min(y)) + min(y)  # 정규화한 값 -> 원본값으로 변경 

                        comp.res[, iline + 1] <- Ym.t.nn[, 1]
                        
                        para.res[iline, 1] <- iline          # Numbering
                        para.res[iline, 2] <- lr[i]          # learning rate(학습률)
                        para.res[iline, 3] <- node.hid[j]    # Num of node in hidden layer(노드 수)
                        para.res[iline, 4] <- epo[k]         # Epoch(학습횟수)
                        para.res[iline, 5] <- 2              # hidden layer(레이어 수)
                        para.res[iline, 6] <- mini.bc[m]     # mini batch
                        para.res[iline, 7] <- gradi.clip[l]  # gradient clip(역전파 가중치 업데이트 작업의 기울기 크기를 제한하는 값)
                        
                        print(paste(iline, '/', allcase, sep = ''))  # 1 ~ 8회까지 실시
                    }
                }   
            }
        }
    }
    return(list(nn.tr, comp.res, para.res))
}


# ------------------------------------------------------------------------------------
# 번  호: 4
# 함수명: makedf_fn
# 입력값: flow_df(유입량에 대한 데이터프레임), df(유역평균강수량, 강수량(지점1 ~ 지점4), 수위(지점1 ~ 지점2)에 대한 데이터프레임)
# 설  명: 사용자가 설정한 홍수예측모델 입력자료 구성 조건에 맞는 데이터프레임 생성(train, test set)
# ------------------------------------------------------------------------------------

makedf_fn <- function(flow_df, df){
    result_df <- 0
    flow_df <- flow_df[, c(flow.start.idx:flow.end.idx)]
    flow_mean_df <- df[, flow.mean.idx]
    rain_region_df <- df[, c(rain.start.idx:rain.end.idx)]
    water_df <- df[, c(water.start.idx:water.end.idx)]
    result_df <- cbind(flow_df, 유역평균강수량 = flow_mean_df, rain_region_df, water_df)
    
    return(result_df)
}


# ------------------------------------------------------------------------------------
# 번  호: 5
# 함수명: plot_fn
# 입력값: actual(실제값), pred(예측값), main(그래프 제목)
# 설  명: 실제값, 예측값 비교 plot
# ------------------------------------------------------------------------------------

plot_fn <- function(actual, pred, main){
    xmin <- 0
    xmax <- max(actual, pred)
    ymin <- 0
    ymax <- max(actual, pred)
    gap <- 1000
    
    plot(-99, -99, type = 'n', xlim = c(xmin, xmax), ylim = c(ymin, ymax), 
         axes = FALSE, cex = 1, mgp = c(2.5, 1, 0), 
         xlab = 'Predicted', ylab = 'Actual', cex.lab = 1, mgp = c(1.9, 1, 0),
         main = main)
        
    tem_gap <- seq(0, 50000, gap)
    abline(h = tem_gap, col = 'gray', lwd = 1, lty = 2)  # 가로 눈금선
    abline(v = tem_gap, col = 'gray', lwd = 1, lty = 2)  # 세로 눈금선
    lines(c(-100:50000), c(-100:50000), col = 'black', lwd = 2, lty = 1)  # 대각선
    points(pred, actual, cex = 1.5, pch = (16), col = 'blue')  # flood events
    axis(1, at = seq(xmin, xmax, by = gap), lab = seq(xmin, xmax, by = gap), lwd.ticks = 1, lwd = 1, cex.axis = 1)
    axis(2, at = seq(ymin, ymax, by = gap), lab = seq(ymin, ymax, by = gap), lwd.ticks = 1, lwd = 1, cex.axis = 1)
    legend('topleft', c('Flood events'), pch = c(16), col = c('blue'), cex = 1.0, bty = 'n')
    box(lwd = 1)
}


# ------------------------------------------------------------------------------------
# 번  호: 6
# 함수명: GOF_twoTS_fn
# 입력값: actual(실제값), pred(예측값)
# 설  명: 적합도 검정을 수행하고, 검정 결과를 출력
# ------------------------------------------------------------------------------------

GOF_twoTS_fn <- function(actual, pred){
    tem <- matrix(NA, length(pred[1, ]), 2)
    for(i in 1:length(pred[1, ])){
        if(sd(pred[, i]) > 0){
            tem_stat <- gof(actual, pred[, i], digits = 4)  # 적합도 검정        
            tem[i, 1] <- tem_stat[4]   # Root mean square error(RMSE)
            tem[i, 2] <- tem_stat[17]  # Coefficient of Determination(R2)
        }
    }
    colnames(tem) <- c('RMSE', 'R2')
    return(tem)
}

