<template>
  <div class="trend-analysis-container">
    <el-row :gutter="20" class="layout">
      <el-col :span="4" class="input-section">
        <el-card class="box-card" shadow="never">
          <template #header>
            <div class="card-header">
              <span>智能分析与可视化输入</span>
              <el-button plain size="small" @click="toggleFullscreen">
                {{ isFullscreen ? '退出' : '全屏' }}
              </el-button>
            </div>
          </template>
          <el-form label-position="top">
            <el-form-item label="临床试验地区">
              <el-radio-group v-model="form.registry">
                <el-radio-button label="中国" value="china" />
                <el-radio-button label="美国" value="usa" />
              </el-radio-group>
            </el-form-item>
            <el-form-item label="选择分析仓库">
              <el-select
                v-model="form.repository"
                placeholder="请选择或输入仓库名称"
                filterable
                allow-create
                default-first-option
                style="width: 100%;"
              >
                <el-option
                  v-for="option in repositoryOptions"
                  :key="option.value"
                  :label="option.label"
                  :value="option.value"
                />
              </el-select>
            </el-form-item>
            <el-form-item label="自然语言需求">
              <el-input
                v-model="form.query"
                type="textarea"
                :rows="6"
                placeholder="例如：帮我分析07年到24年到临床试验免疫治疗的趋势，关于免疫治疗包括但不限于免疫联合其他治疗，免疫+靶向治疗，免疫+局部治疗等等免疫联合治疗的趋势"
              />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :loading="loading" class="submit-btn" @click="handleGenerate">
                生成分析与可视化
              </el-button>
            </el-form-item>
          </el-form>
        </el-card>
      </el-col>

      <el-col :span="20" class="result-section">
        <div ref="resultPaneRef" class="result-pane">
          <div class="charts-wrapper" v-loading="loading">
          <el-row :gutter="12">
            <el-col :span="24">
              <el-card class="chart-card" shadow="never">
                <template #header>
                  <div class="card-header">
                    <span>历年注册试验数量趋势分析</span>
                    <el-tag v-if="generatedAt" type="success">已按条件刷新</el-tag>
                  </div>
                </template>
                <iframe :src="chartSources.trend" class="chart-iframe trend-iframe"></iframe>
                <div class="chart-analysis">
                  <el-alert :title="getInsight('trend').summary" type="success" :closable="false" show-icon />
                  <ul class="analysis-list">
                    <li v-for="(item, index) in getInsight('trend').bullets" :key="`trend-${index}`">{{ item }}</li>
                  </ul>
                </div>
              </el-card>
            </el-col>
          </el-row>


          <el-row :gutter="12" class="section-gap">
            <el-col :span="24">
              <el-card class="chart-card" shadow="never">
                <template #header>
                  <div class="card-header">
                    <span>全球临床试验分布</span>
                  </div>
                </template>
                <iframe :src="chartSources.map" class="chart-iframe map-iframe"></iframe>
                <div class="chart-analysis">
                  <el-alert :title="getInsight('map').summary" type="success" :closable="false" show-icon />
                  <ul class="analysis-list">
                    <li v-for="(item, index) in getInsight('map').bullets" :key="`map-${index}`">{{ item }}</li>
                  </ul>
                </div>
              </el-card>
            </el-col>
          </el-row>

          <el-row :gutter="12" class="section-gap">
            <el-col :span="24">
              <el-card class="chart-card" shadow="never">
                <template #header>
                  <div class="card-header">
                    <span>不同治疗方式的年度注册数量</span>
                  </div>
                </template>
                <iframe :src="chartSources.treatment" class="chart-iframe treatment-iframe"></iframe>
                <div class="chart-analysis">
                  <el-alert :title="getInsight('treatment').summary" type="success" :closable="false" show-icon />
                  <ul class="analysis-list">
                    <li v-for="(item, index) in getInsight('treatment').bullets" :key="`treatment-${index}`">{{ item }}</li>
                  </ul>
                </div>
              </el-card>
            </el-col>
          </el-row>

          <el-row :gutter="12" class="section-gap">
            <el-col :span="12">
              <el-card class="chart-card" shadow="never">
                <template #header>
                  <div class="card-header">
                    <span>主要分类变量权重分析</span>
                  </div>
                </template>
                <iframe :src="chartSources.weights" class="chart-iframe short"></iframe>
                <div class="chart-analysis">
                  <el-alert :title="getInsight('weights').summary" type="success" :closable="false" show-icon />
                  <ul class="analysis-list">
                    <li v-for="(item, index) in getInsight('weights').bullets" :key="`weights-${index}`">{{ item }}</li>
                  </ul>
                </div>
              </el-card>
            </el-col>
            <el-col :span="12">
              <el-card class="chart-card" shadow="never">
                <template #header>
                  <div class="card-header">
                    <span>试验失败原因分布</span>
                  </div>
                </template>
                <iframe :src="chartSources.failure" class="chart-iframe short"></iframe>
                <div class="chart-analysis">
                  <el-alert :title="getInsight('failure').summary" type="success" :closable="false" show-icon />
                  <ul class="analysis-list">
                    <li v-for="(item, index) in getInsight('failure').bullets" :key="`failure-${index}`">{{ item }}</li>
                  </ul>
                </div>
              </el-card>
            </el-col>
          </el-row>

          <el-row :gutter="12" class="section-gap">
            <el-col :span="24">
              <el-card class="chart-card" shadow="never">
                <template #header>
                  <div class="card-header">
                    <span>各个类别生存分析</span>
                  </div>
                </template>
                <iframe :src="chartSources.survival" class="chart-iframe"></iframe>
                <div class="chart-analysis">
                  <el-alert :title="getInsight('survival').summary" type="success" :closable="false" show-icon />
                  <ul class="analysis-list">
                    <li v-for="(item, index) in getInsight('survival').bullets" :key="`survival-${index}`">{{ item }}</li>
                  </ul>
                </div>
              </el-card>
            </el-col>
          </el-row>
          </div>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue';
import { ElMessage } from 'element-plus';

type RegistryType = 'china' | 'usa';
type InsightKey = 'trend' | 'treatment' | 'map' | 'weights' | 'failure' | 'survival';
interface ChartInsight {
  summary: string;
  bullets: string[];
}

const repositoryOptions = [
  { label: '肝癌专题仓', value: 'live-oncology-repo' },
  { label: '实体瘤联合疗法仓', value: 'solid-tumor-combo-repo' },
  { label: '血液肿瘤仓', value: 'hematology-repo' }
];

const form = reactive({
  registry: 'china' as RegistryType,
  repository: 'live-oncology-repo',
  query: ''
});

const loading = ref(false);
const generatedAt = ref(0);
const isFullscreen = ref(false);
const resultPaneRef = ref<HTMLElement | null>(null);
const fallbackInsight: ChartInsight = {
  summary: '请先生成分析与可视化',
  bullets: ['系统将在图表刷新后生成基于真实图表数据的分析结论。']
};
const chartInsights = ref<Record<InsightKey, ChartInsight>>({
  trend: fallbackInsight,
  treatment: fallbackInsight,
  map: fallbackInsight,
  weights: fallbackInsight,
  failure: fallbackInsight,
  survival: fallbackInsight
});

const chartSources = computed(() => {
  const query = new URLSearchParams({
    mock: '1',
    registry: form.registry,
    repository: form.repository.trim(),
    query: form.query.trim(),
    t: String(generatedAt.value)
  }).toString();
  return {
    trend: `http://localhost:8000/api/analysis/trend?${query}`,
    treatment: `http://localhost:8000/api/analysis/treatment?${query}`,
    map: `http://localhost:8000/api/analysis/map?${query}`,
    weights: `http://localhost:8000/api/analysis/weights?${query}`,
    failure: `http://localhost:8000/api/analysis/failure?${query}`,
    survival: `http://localhost:8000/api/analysis/survival?${query}`
  };
});

const getInsight = (key: InsightKey) => chartInsights.value[key] || fallbackInsight;

const loadInsights = async (queryString: string) => {
  const response = await fetch(`http://localhost:8000/api/analysis/insights?${queryString}`);
  if (!response.ok) {
    throw new Error('insights request failed');
  }
  const data = await response.json();
  if (data?.code !== 200 || !data?.data) {
    throw new Error(data?.msg || 'insights payload invalid');
  }
  const insightData = data.data as Partial<Record<InsightKey, ChartInsight>>;
  chartInsights.value = {
    trend: insightData.trend || fallbackInsight,
    treatment: insightData.treatment || fallbackInsight,
    map: insightData.map || fallbackInsight,
    weights: insightData.weights || fallbackInsight,
    failure: insightData.failure || fallbackInsight,
    survival: insightData.survival || fallbackInsight
  };
};

const handleGenerate = async () => {
  if (!form.repository.trim()) {
    ElMessage.warning('请先选择或输入分析仓库');
    return;
  }
  if (!form.query.trim()) {
    ElMessage.warning('请输入自然语言分析需求');
    return;
  }

  loading.value = true;
  const queryString = new URLSearchParams({
    mock: '1',
    registry: form.registry,
    repository: form.repository.trim(),
    query: form.query.trim()
  }).toString();
  try {
    await Promise.all([
      new Promise((resolve) => {
        setTimeout(resolve, 500);
      }),
      loadInsights(queryString)
    ]);
  } catch (error) {
    console.error(error);
    ElMessage.warning('图表分析文本生成失败，已保留图表刷新结果');
  }
  generatedAt.value = Date.now();
  loading.value = false;
  ElMessage.success('已刷新图表与逐图分析结果');
};

const syncFullscreenState = () => {
  isFullscreen.value = document.fullscreenElement === resultPaneRef.value;
};

const toggleFullscreen = async () => {
  if (!resultPaneRef.value) return;
  try {
    if (document.fullscreenElement === resultPaneRef.value) {
      await document.exitFullscreen();
      return;
    }
    if (document.fullscreenElement) {
      await document.exitFullscreen();
    }
    await resultPaneRef.value.requestFullscreen();
  } catch (error) {
    console.error(error);
    ElMessage.error('当前环境暂不支持该全屏操作');
  }
};

onMounted(() => {
  document.addEventListener('fullscreenchange', syncFullscreenState);
});

onBeforeUnmount(() => {
  document.removeEventListener('fullscreenchange', syncFullscreenState);
});
</script>

<style scoped>
.trend-analysis-container {
  padding: 20px;
  box-sizing: border-box;
}

.layout {
  height: calc(100vh - 120px);
}

.input-section,
.result-section {
  height: 100%;
}

.box-card {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.chart-card {
  width: 100%;
}

:deep(.el-card__body) {
  flex: 1;
  overflow-y: auto;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.submit-btn {
  width: 100%;
  margin-top: 10px;
}

.result-pane {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.charts-wrapper {
  flex: 1;
  overflow-y: auto;
  padding-right: 2px;
}

.section-gap {
  margin-top: 12px;
}

.chart-iframe {
  width: 100%;
  height: 460px;
  border: none;
  overflow: hidden;
}

.chart-iframe.short {
  height: 500px;
}

.chart-iframe.map-iframe {
  height: 560px;
}

.chart-iframe.trend-iframe,
.chart-iframe.treatment-iframe {
  height: 540px;
}

.result-pane:fullscreen {
  background: #fff;
  padding: 12px;
  box-sizing: border-box;
}

.chart-analysis {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 10px;
}

.analysis-list {
  margin: 0;
  padding-left: 18px;
  color: #606266;
  line-height: 1.8;
}
</style>
