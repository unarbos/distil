export interface Teacher {
  model: string;
  totalParams: number;
  activeParams: number;
  vocabSize: number;
  architecture: string;
  maxStudentParams: number;
}

export interface SubnetConfig {
  netuid: number | null;
  maxKlThreshold: number;
  emaAlpha: number;
  maxNewTokens: number;
  maxPromptTokens: number;
  samplesPerEpoch: number;
}
