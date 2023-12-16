T1_name = 't1_wr41_Square_acc_sr1_randinit_ppo_';
T2_name = 't1_wr41_Square_acc_sr1_randinit_sac_';
T3_name = 'no_ctrl_0init';
T4_name = 'no_ctrl_1init';
T5_name = 't1_wr41_Square_acc_sr1_0init_ppo_';
T6_name = 't1_wr41_Square_acc_sr2_0init_ppo_';

T1 = readtable([T1_name,'.csv']);
T2 = readtable([T2_name,'.csv']);
T3 = readtable([T3_name,'.csv']);
T4 = readtable([T4_name,'.csv']);
T5 = readtable([T5_name,'.csv']);
T6 = readtable([T6_name,'.csv']);


drawT1 = 0  ;
drawT2 = 0  ;
drawT3 = 1  ;
drawT4 = 1 ;
drawT5 = 0  ;
drawT6 = 0  ;

T1_name = 't1_wr41_Square_acc_sr1_randinit_ppo_';
T2_name = 't1_wr41_Square_acc_sr1_randinit_sac_';
T3_name = 'Init: 0 0 0 0';
T4_name = 'Init: 0.5 0 0 0';
T5_name = 't1_wr41_Square_acc_sr1_0init_ppo_';
T6_name = 't1_wr41_Square_acc_sr2_0init_ppo_';

%% 
figure(1)

subplot(2,2,1)
if drawT1
    plot(T1.x1,'Linewidth',2,'DisplayName',T1_name);
end
hold on
if drawT2
    plot(T2.x1,'Linewidth',2,'DisplayName',T2_name);
end
if drawT3
    plot(T3.x1,'Linewidth',2,'DisplayName',T3_name);
end
if drawT4
    plot(T4.x1,'Linewidth',2,'DisplayName',T4_name);
end
if drawT5
    plot(T5.x1,'Linewidth',2,'DisplayName',T5_name);
end
if drawT6
    plot(T6.x1,'Linewidth',2,'DisplayName',T6_name);
end
title('x1')
legend()
grid on
xlim([0,300])
set(gca, 'linewidth', 1.1, 'fontsize', 17, 'fontname', 'times','color','#E6E8E9') %去掉x，y坐标轴的刻度

subplot(2,2,2)
if drawT1
    plot(T1.x2,'Linewidth',2,'DisplayName',T1_name);
end
hold on
if drawT2
    plot(T2.x2,'Linewidth',2,'DisplayName',T2_name);
end
if drawT3
    plot(T3.x2,'Linewidth',2,'DisplayName',T3_name);
end
if drawT4
    plot(T4.x2,'Linewidth',2,'DisplayName',T4_name);
end
if drawT5
    plot(T5.x2,'Linewidth',2,'DisplayName',T5_name);
end
if drawT6
    plot(T6.x2,'Linewidth',2,'DisplayName',T6_name);
end
title('x2')
%legend()
grid on
xlim([0,300])
set(gca, 'linewidth', 1.1, 'fontsize', 17, 'fontname', 'times','color','#E6E8E9') %去掉x，y坐标轴的刻度

subplot(2,2,3)
if drawT1
    plot(T1.x3,'Linewidth',2,'DisplayName',T1_name);
end
hold on
if drawT2
    plot(T2.x3,'Linewidth',2,'DisplayName',T2_name);
end
if drawT3
    plot(T3.x3,'Linewidth',2,'DisplayName',T3_name);
end
if drawT4
    plot(T4.x3,'Linewidth',2,'DisplayName',T4_name);
end
if drawT5
    plot(T5.x3,'Linewidth',2,'DisplayName',T5_name);
end
if drawT6
    plot(T6.x3,'Linewidth',2,'DisplayName',T6_name);
end
title('x3')
%legend()
grid on
xlim([0,300])
 ylim([-0.5,0.5])
set(gca, 'linewidth', 1.1, 'fontsize', 17, 'fontname', 'times','color','#E6E8E9') %去掉x，y坐标轴的刻度


subplot(2,2,4)
if drawT1
    plot(T1.x4,'Linewidth',2,'DisplayName',T1_name);
end
hold on
if drawT2
    plot(T2.x4,'Linewidth',2,'DisplayName',T2_name);
end
if drawT3
    plot(T3.x4,'Linewidth',2,'DisplayName',T3_name);
end
if drawT4
    plot(T4.x4,'Linewidth',2,'DisplayName',T4_name);
end
if drawT5
    plot(T5.x4,'Linewidth',2,'DisplayName',T5_name);
end
if drawT6
    plot(T6.x4,'Linewidth',2,'DisplayName',T6_name);
end
title('x4')
%legend()
grid on
xlim([0,300])
 ylim([-0.5,0.5])
set(gca, 'linewidth', 1.1, 'fontsize', 17, 'fontname', 'times','color','#E6E8E9') %去掉x，y坐标轴的刻度

