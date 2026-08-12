一：状态机情况说明: //本文档不将LocalObservation和ProjectedObservation区分
   （1）每一个任务（也就是对一个目标的连续追踪任务）有唯一 state machine ，输入初始化必须数据，输出为 TrackingResult(包含每一帧的预测框的List，如有其他必须信息课补充)
   （2）state 的类型有 init,tracking,uncertain,recovery,lost ，将状态机每次进入一个状态定义为一个state (tracking -> trakcking的切换这两个tracking也是不同state)
   （3）每一个 state 有自己的ViewSpec要求，且一次状态对应唯一ViewSpec实例
   （4）每一个 state 对应的ViewSpec每张图片经过后端会返回一个LocalObservation（包括图片id，局部框，预测分数等信息)
   （5）LocalObservation list和ViewSpec图片一一对应（ViewSpec中不只是照片数据这里只是简称为照片，不修改目前ViewSpec格式）
   （6）每一个state根据他的LocalObservation list信息通过StateEvaluator处理成一个StateObservation(注意原来的FrameAggregate.bbox修改为StateObservation.bbox，后面介绍包含信息列表)
   （7）state根据他的StateObservation进行状态切换的选择和判断
   （8）每一个 state 有自己的一个 PredictedCenter, 也就是在刚进入该状态还未给出ViewSpec之前，多帧运动预测中心给出，除了从 Lost 返回的tracking和uncertain

二：StateObservation 包含数据（暂定，如有其他建议信息，可补充）
    1.state状态: init,tracking,uncertain,recovery,lost
    2.预测框信息: 也就是根据LocalObservation List 本帧预测框的中心宽高等和之前实现一致
    3.StateScore: 本state预测框判定的位置的预测准确度（单独算法计算帮我实现）
    4.EverLost: bool值,初始化为0,从Lost返回uncertain或者tracking的时候置1，避免同一帧反复进入Lost卡死
    5.ResultCenter: 预测框的中心，不同于PredictedCenter,且多帧运动预测模块应当根据前几帧的ResultCenter预测而非PredictedCenter!!!

三：状态切换:(在本部分第零帧的init仍称为init,但是第n帧的tracking命为tracking[n],同理uncertain[n],lost[n],仅表明该状态处理哪一帧内容)
  
  (1)init:
  init ->tracking[1]         //初始化，初始化后到tracking
  
  (2)tracking:
  tracking[n] -> output   if score >= LostThreshold   //结束后，将TrackingResult输出(最后一帧也可以进入一次lost)
  tracking[n] -> tracking[n+1]  if  score >= ReliableThreshold    //ReliableThreshold是超参数
  tracking[n] -> uncertain[n+1]  if  ReliableThreshold > score >= LostThreshold   //进入uncertain但是仍然到下一帧,LostThreshold也是超参数
  tracking[n] -> lost[n]  if score < LostThreshold  //丢失停留在本帧
  
  (3)uncertain:
  uncertain[n] -> output  if score >= LostThreshold
  uncertain[n] -> lost[n] if score < LostThreshold  //丢失进入Lost
  uncertain[n] -> tracking [n+1]  if score > ReliableThreshold   //预测成功回归正常状态
  uncertain[n] -> recovery[n+1]  if ReliableThreshold > score >= LostThreshold  //连续两次score较低，进入找回模式
  
  (4)lost:   
    //lost返回去的tracking[n]或uncertain[n]，将Lost的ResultCenter，作为后者的PredictedCenter重新进入ViewSpec阶段进行重新获得（相当于Lost帮他们重新找物体中心）
    lost[n] -> uncertain[n] if score < ReliableThreshold and 将新的uncertain[n]的EverLost置1
    lost[n] -> tracking[n] if score >= ReliableThreshold and 将新的uncertain[n]的EverLost置1
  
  (5)recovery: 
   recovery[n] -> output
   recovery[n] -> tracking[n+1]  if  score >= ReliableThreshold
   recovery[n] -> uncertain[n+1] if  score < ReliableThreshold

四：每个的SpecView:  //注意：现在开始要求，每一张SpecView要求的图片大小都是localView能占球面覆盖比例的最大大小，即均使用最大视场！！！
   （1）tracking 和 uncertain 状态的ViewSpec需要 5 张图 1.以 PredictedCenter 为中心的图一 2.以图一的视场四个角为中心，分别再做四张图
   （2）lost 和 recovery 需要 6 张图，以 cube-map布置的六张图

五: 每个状态内部线程 
   （一）非Lost返回
    切换至本状态 -> 多帧运动预测模块计算出 PredictedCenter -> 计算出对应的SpecView -> SpecView经过后端给出 LocalObservation list 
    -> 经过 StateEvaluator 得到 StateObservation -> （非Lost，Lost不更新）根据StateObservation 更新 TrackingResult -> 切换下一状态
   （二）Lost返回(返回的uncertain[n]或tracking[n]和之前的是同一对象，只是根据上一轮更新)
    切换至本状态 -> 多帧运动预测模块中根据Lost给出的 ResultCenter 作为 PredictedCenter -> 计算出新的的SpecView -> SpecView经过后端给出新的 LocalObservation list
    -> 经过 StateEvaluator 得到新的 StateObservation -> 根据StateObservation 更新 本帧 TrackingResult（覆盖进入Lost之前给出的）-> 切换下一帧

六: 状态机辅助工具（一）多帧运动预测模块
    1.如果非从Lost返回 根据前几帧的 ResultCenter 进行预测（算法同之前可适当优化）
    2.如果从Lost返回 将Lost给出的 ResultCenter 作为 PredictedCenter

七: 状态机辅助工具 (二) StateEvaluator //用于处理localObservation List并返回StateObservation
    1.注意使用的是在球体上的坐标
    2.由于多个视场不在同一画面中，如果没有重合，（1）如果两个LocalObservation有交叉，选择最大并集矩形作为输出 （2）如果没有相交，直接选择置信度最高的LocalObservation位置作为输出
