clc,clear
T1_name = 'train/FTask1_wr41_Square_acc_sr005_0init_xtheta_ppo_TranslationOscillatorEnv-v0_1';
T2_name = 'train/FTask1_wr41_Square_acc_sr01_0init_xtheta_ppo_TranslationOscillatorEnv-v0_1';
T3_name = 'train/FTask1_wr41_Square_acc_sr02_0init_xtheta_ppo_TranslationOscillatorEnv-v0_1';
T4_name = 'no_ctrl_0init';
T5_name = 't1_wr41_Square_acc_sr1_0init_ppo_';
T6_name = 't1_wr41_Square_acc_sr2_0init_ppo_';

T1 = readtable([T1_name,'.csv']);
T2 = readtable([T2_name,'.csv']);
T3 = readtable([T3_name,'.csv']);
T4 = readtable([T4_name,'.csv']);
T5 = readtable([T5_name,'.csv']);
T6 = readtable([T6_name,'.csv']);


drawT1 = 1  ;
drawT2 = 1  ;
drawT3 = 1  ;
drawT4 = 0 ;
drawT5 = 0  ;
drawT6 = 0  ;

T1_name = 'a_s: 0.005 ';
T2_name = 'a_s: 0.01';
T3_name = 'a_s: 0.02';
T4_name = 'No control';
T5_name = 't1_wr41_Square_acc_sr1_0init_ppo_';
T6_name = 't1_wr41_Square_acc_sr2_0init_ppo_';

%% 
figure(1)

plot(T1.Step,T1.Value,'Linewidth',2,'DisplayName',T1_name);

hold on

plot(T2.Step,T2.Value,'Linewidth',2,'DisplayName',T2_name);


plot(T3.Step,T3.Value,'Linewidth',2,'DisplayName',T3_name);

xlabel('timesteps')
ylabel('rollout reward')
title('零状态稳定控制PPO训练结果')
legend()
grid on
xlim([0  3100000])
set(gca, 'linewidth', 1.1, 'fontsize', 13,'color','#E6E8E9') %去掉x，y坐标轴的刻度



