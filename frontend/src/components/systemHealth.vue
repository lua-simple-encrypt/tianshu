<template>
  <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
    <div class="flex items-center justify-between mb-4">
      <h3 class="text-sm font-bold text-gray-700 flex items-center">
        <Activity class="w-4 h-4 mr-2 text-primary-500" />
        系统集群状态
      </h3>
      <span class="text-[10px] text-gray-400">每 10s 自动刷新</span>
    </div>

    <div class="space-y-3">
      <div v-for="(status, name) in services" :key="name" 
           class="flex items-center justify-between px-3 py-2 rounded-lg bg-gray-50 border border-gray-100">
        <div class="flex flex-col">
          <span class="text-[11px] font-bold text-gray-500 uppercase tracking-wider">{{ formatLabel(name) }}</span>
          <span class="text-[10px] text-gray-400">{{ getUrlHint(name) }}</span>
        </div>
        <div class="flex items-center">
          <span :class="['w-2.5 h-2.5 rounded-full mr-2 shadow-sm', getStatusBg(status)]" 
                class="relative">
            <span v-if="status === 'online'" 
                  class="absolute inset-0 rounded-full bg-inherit animate-ping opacity-75"></span>
          </span>
          <span :class="['text-xs font-semibold', getStatusText(status)]">
            {{ status.toUpperCase() }}
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { Activity } from 'lucide-vue-next';

defineProps<{ services: Record<string, string> }>();

const formatLabel = (key: string) => {
  const map: Record<string, string> = {
    database: '任务数据库',
    local_worker: '本地 GPU 解析器',
    vllm_paddle_8118: 'Paddle-VLM 集群',
    vllm_mineru_8119: 'MinerU-VLM 集群'
  };
  return map[key] || key;
};

const getUrlHint = (key: string) => {
  if (key.includes('8118')) return 'Port: 8118';
  if (key.includes('8119')) return 'Port: 8119';
  if (key.includes('worker')) return 'Internal Worker';
  return 'Local Storage';
};

const getStatusBg = (s: string) => s === 'online' ? 'bg-green-500' : (s === 'offline' ? 'bg-red-500' : 'bg-yellow-500');
const getStatusText = (s: string) => s === 'online' ? 'text-green-600' : (s === 'offline' ? 'text-red-600' : 'text-yellow-600');
</script>
