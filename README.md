# SmartYieldTool
# 项目架构图

## 整体架构
```mermaid
flowchart TD
    A[main.py\n主程序入口\nYieldAnalysisApp类\nTkinter界面] -->|调用| B[analysis.py\nDataAnalyzer类\n数据分析逻辑]
    B -->|返回分析结果| A
    subgraph main.py
        A1[界面初始化/文件选择]
        A2[参数设置/分析方法选择]
        A3[调用DataAnalyzer进行分析]
        A4[结果展示/导出]
        A1 --> A2 --> A3 --> A4
    end
    subgraph analysis.py
        B1[文件读取/预处理]
        B2[极限值计算/分布分析]
        B3[报告生成/推荐]
        B1 --> B2 --> B3
    end
    A3 -->|传递文件、参数| B1
    B3 -->|分析结果/报告| A4
```

## 数据分析详细架构
```mermaid
flowchart TD
    DA[DataAnalyzer类]
    RF[读取文件]
    CL[单列极限值]
    CLC[多列极限值]
    AD[分布分析]
    SRL[智能推荐上下限]
    SRLC[批量智能推荐]
    GR[生成报告]
    DA --> RF
    DA --> CL
    DA --> CLC
    DA --> AD
    DA --> SRL
    DA --> SRLC
    DA --> GR
    RF --> CL
    RF --> CLC
    RF --> AD
    RF --> SRL
    RF --> SRLC
    CL --> GR
    CLC --> GR
    SRL --> GR
    SRLC --> GR
    AD --> SRL
    AD --> SRLC
```

> 说明：  
> - 读取文件：_read_file  
> - 单列极限值：calculate_limits  
> - 多列极限值：calculate_limits_for_columns  
> - 分布分析：analyze_distribution  
> - 智能推荐上下限：smart_recommend_limits  
> - 批量智能推荐：smart_recommend_limits_for_columns  
> - 生成报告：generate_report  
