
export interface Prediction {
  id: string;
  model: string;
  output: string;
  confidence: number;
  processed: string;
  status: 'win' | 'loss' | 'high-risk';
}

export interface Metric {
  label: string;
  value: string;
  progress: number;
}

export interface Champion {
  name: string;
  role: string;
  kda: number;
  image: string;
}

export interface MatchHistoryItem {
  result: 'Victory' | 'Defeat';
  mode: string;
  duration: string;
  champion: Champion;
  score: string;
  kda: string;
  scoreValue: number;
  rank: 'MVP' | 'ACE' | 'S+';
}
