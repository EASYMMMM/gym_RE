T1 = readtable('t1_Square_ppo_.csv');
T2 = readtable('t1_NoSquare_ppo_.csv');
T3 = readtable('t1_Square_sac_.csv');
T4 = readtable('t1_NoSquare_sac_.csv');
T5 = readtable('t1_wr41_Square_ppo_.csv');
T6 = readtable('t1_wr41_NoSquare_ppo_.csv');


drawT1 = 1;
drawT2 = 1;
drawT3 = 1;
drawT4 = 0;
drawT5 = 1;
drawT6 = 0;
%% 
figure(1)
subplot(2,2,1)
if drawT1
    plot(T1.x1,'Linewidth',2,'DisplayName','Square ppo');
end
hold on
if drawT2
    plot(T2.x1,'Linewidth',2,'DisplayName','NoSquare ppo');
end
if drawT3
    plot(T3.x1,'Linewidth',2,'DisplayName','Square sac');
end
if drawT4
    plot(T4.x1,'Linewidth',2,'DisplayName','NoSquare sac');
end
if drawT5
    plot(T5.x1,'Linewidth',2,'DisplayName','wr41 Square ppo');
end
if drawT6
    plot(T6.x1,'Linewidth',2,'DisplayName','wr41 NoSquare ppo');
end
title('x1')
legend()
grid on

subplot(2,2,2)
if drawT1
    plot(T1.x2,'Linewidth',2,'DisplayName','Square ppo');
end
hold on
if drawT2
    plot(T2.x2,'Linewidth',2,'DisplayName','NoSquare ppo');
end
if drawT3
    plot(T3.x2,'Linewidth',2,'DisplayName','Square sac');
end
if drawT4
    plot(T4.x2,'Linewidth',2,'DisplayName','NoSquare sac');
end
if drawT5
    plot(T5.x2,'Linewidth',2,'DisplayName','wr41 Square ppo');
end
if drawT6
    plot(T6.x2,'Linewidth',2,'DisplayName','wr41 NoSquare ppo');
end
title('x2')
legend()
grid on


subplot(2,2,3)
if drawT1
    plot(T1.x3,'Linewidth',2,'DisplayName','Square ppo');
end
hold on
if drawT2
    plot(T2.x3,'Linewidth',2,'DisplayName','NoSquare ppo');
end
if drawT3
    plot(T3.x3,'Linewidth',2,'DisplayName','Square sac');
end
if drawT4
    plot(T4.x3,'Linewidth',2,'DisplayName','NoSquare sac');
end
if drawT5
    plot(T5.x3,'Linewidth',2,'DisplayName','wr41 Square ppo');
end
if drawT6
    plot(T6.x3,'Linewidth',2,'DisplayName','wr41 NoSquare ppo');
end
title('x3')
legend()
grid on

subplot(2,2,4)
if drawT1
    plot(T1.x4,'Linewidth',2,'DisplayName','Square ppo');
end
hold on
if drawT2
    plot(T2.x4,'Linewidth',2,'DisplayName','NoSquare ppo');
end
if drawT3
    plot(T3.x4,'Linewidth',2,'DisplayName','Square sac');
end
if drawT4
    plot(T4.x4,'Linewidth',2,'DisplayName','NoSquare sac');
end
if drawT5
    plot(T5.x4,'Linewidth',2,'DisplayName','wr41 Square ppo');
end
if drawT6
    plot(T6.x4,'Linewidth',2,'DisplayName','wr41 NoSquare ppo');
end
title('x4')
legend()
grid on