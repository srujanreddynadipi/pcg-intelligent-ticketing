import { useState, useCallback } from 'react';
import { mlApiClient } from '@/lib/api';
import { MLPredictResponse, HistoricalTicket, KnowledgeBaseArticle } from '@/lib/types';

export interface MLPredictPayload {
  user: string; // email
  title: string;
  description: string;
  historical_tickets: HistoricalTicket[];
  knowledge_base: KnowledgeBaseArticle[];
}

export const useMLPredict = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const predict = useCallback(
    async (payload: MLPredictPayload): Promise<MLPredictResponse | null> => {
      setIsLoading(true);
      setError(null);
      try {
        console.log('[ML] Sending prediction request with payload:', payload);
        
        const response = await mlApiClient.predict<MLPredictResponse>(payload);
        console.log('[ML] Prediction response:', response);
        
        if (response.ticket_id && response.predictions) {
          return response;
        } else {
          throw new Error('No ML prediction in response');
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'ML prediction failed';
        console.error('[ML] Prediction error:', errorMessage);
        setError(errorMessage);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  return {
    predict,
    isLoading,
    error,
  };
};
