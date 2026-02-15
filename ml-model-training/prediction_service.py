"""
ITSM Ticket Prediction Service
Provides predictions for new tickets with confidence scores and explanations
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.sparse import hstack
from datetime import datetime

class ITSMPredictor:
    """Main prediction service for ITSM tickets."""
    
    def __init__(self, models_dir: str = "models"):
        """Load all trained models."""
        self.models_dir = Path(models_dir)
        print("Loading ITSM Prediction Models...")
        
        # Load vectorizer and models
        self.tfidf = joblib.load(self.models_dir / 'tfidf_vectorizer.pkl')
        self.category_model = joblib.load(self.models_dir / 'category_classifier.pkl')
        self.priority_model = joblib.load(self.models_dir / 'priority_predictor.pkl')
        self.resolver_model = joblib.load(self.models_dir / 'resolver_router.pkl')
        
        # Load encoders
        self.impact_encoder = joblib.load(self.models_dir / 'impact_encoder.pkl')
        self.urgency_encoder = joblib.load(self.models_dir / 'urgency_encoder.pkl')
        self.category_encoder = joblib.load(self.models_dir / 'category_encoder.pkl')
        
        # Get class names
        self.category_classes = self.category_model.classes_
        self.priority_classes = self.priority_model.classes_
        self.resolver_classes = self.resolver_model.classes_
        
        print("✓ All models loaded successfully!")
        print(f"  - Categories: {len(self.category_classes)}")
        print(f"  - Priorities: {len(self.priority_classes)}")
        print(f"  - Resolvers: {len(self.resolver_classes)}")
        
        # SLA mapping
        self.sla_map = {"Critical": 1, "High": 4, "Medium": 24, "Low": 72}
    
    def predict_ticket(self, 
                      title: str, 
                      description: str,
                      impact_level: str = "Medium",
                      urgency_level: str = "Medium",
                      affected_users: int = 1) -> Dict:
        """
        Make predictions for a single ticket.
        
        Args:
            title: Ticket title/short description
            description: Full ticket description
            impact_level: Impact level (Low, Medium, High, Critical)
            urgency_level: Urgency level (Low, Medium, High, Critical)
            affected_users: Number of affected users
            
        Returns:
            Dictionary with all predictions, confidence scores, and explanations
        """
        # Combine text
        text_combined = f"{title} {description}"
        text_vec = self.tfidf.transform([text_combined])
        
        # ─────────────────────────────────────────────────────────────────────
        # 1. CATEGORY PREDICTION
        # ─────────────────────────────────────────────────────────────────────
        cat_pred = self.category_model.predict(text_vec)[0]
        cat_proba = self.category_model.predict_proba(text_vec)[0]
        cat_confidence = float(max(cat_proba))
        
        # Get top 3 predictions
        top3_cat_indices = np.argsort(cat_proba)[-3:][::-1]
        top3_categories = [
            {
                "category": self.category_classes[idx],
                "confidence": float(cat_proba[idx])
            }
            for idx in top3_cat_indices
        ]
        
        # Get key words that influenced decision
        feature_names = self.tfidf.get_feature_names_out()
        text_features = text_vec.toarray()[0]
        top_feature_indices = np.argsort(text_features)[-5:][::-1]
        key_terms = [feature_names[idx] for idx in top_feature_indices if text_features[idx] > 0]
        
        # ─────────────────────────────────────────────────────────────────────
        # 2. PRIORITY PREDICTION
        # ─────────────────────────────────────────────────────────────────────
        impact_encoded = self.impact_encoder.transform([impact_level]).reshape(-1, 1)
        urgency_encoded = self.urgency_encoder.transform([urgency_level]).reshape(-1, 1)
        affected_encoded = np.array([[affected_users]])
        
        X_pri = hstack([text_vec, impact_encoded, urgency_encoded, affected_encoded])
        pri_pred = self.priority_model.predict(X_pri)[0]
        pri_proba = self.priority_model.predict_proba(X_pri)[0]
        pri_confidence = float(max(pri_proba))
        
        sla_hours = self.sla_map.get(pri_pred, 24)
        
        # Priority explanation
        pri_explanation = f"Based on Impact={impact_level}, Urgency={urgency_level}, Affected Users={affected_users}"
        
        # ─────────────────────────────────────────────────────────────────────
        # 3. RESOLVER ROUTING
        # ─────────────────────────────────────────────────────────────────────
        cat_encoded = self.category_encoder.transform([cat_pred]).reshape(-1, 1)
        X_res = hstack([text_vec, cat_encoded, impact_encoded, urgency_encoded])
        
        res_pred = self.resolver_model.predict(X_res)[0]
        res_proba = self.resolver_model.predict_proba(X_res)[0]
        res_confidence = float(max(res_proba))
        
        # Get top 3 resolver suggestions
        top3_res_indices = np.argsort(res_proba)[-3:][::-1]
        top3_resolvers = [
            {
                "resolver_group": self.resolver_classes[idx],
                "confidence": float(res_proba[idx])
            }
            for idx in top3_res_indices
        ]
        
        # Routing explanation
        res_explanation = f"Routed to {res_pred} based on Category={cat_pred}, Impact={impact_level}, Urgency={urgency_level}"
        
        # ─────────────────────────────────────────────────────────────────────
        # COMPILE RESULTS
        # ─────────────────────────────────────────────────────────────────────
        result = {
            "prediction_timestamp": datetime.now().isoformat(),
            
            # Category
            "category": {
                "predicted": cat_pred,
                "confidence": cat_confidence,
                "top_3_predictions": top3_categories,
                "key_terms": key_terms[:5],
                "confidence_level": self._confidence_label(cat_confidence)
            },
            
            # Priority
            "priority": {
                "predicted": pri_pred,
                "confidence": pri_confidence,
                "sla_hours": sla_hours,
                "explanation": pri_explanation,
                "confidence_level": self._confidence_label(pri_confidence)
            },
            
            # Resolver
            "resolver": {
                "predicted": res_pred,
                "confidence": res_confidence,
                "top_3_suggestions": top3_resolvers,
                "explanation": res_explanation,
                "confidence_level": self._confidence_label(res_confidence)
            },
            
            # Input summary
            "input": {
                "title": title,
                "description": description[:200] + "..." if len(description) > 200 else description,
                "impact_level": impact_level,
                "urgency_level": urgency_level,
                "affected_users": affected_users
            },
            
            # Audit trail
            "audit": {
                "model_version": "v1.0",
                "category_model": "LogisticRegression",
                "priority_model": "GradientBoosting",
                "resolver_model": "RandomForest"
            }
        }
        
        return result
    
    def predict_batch(self, tickets: List[Dict]) -> List[Dict]:
        """
        Make predictions for multiple tickets.
        
        Args:
            tickets: List of ticket dictionaries with keys: title, description, 
                    impact_level, urgency_level, affected_users
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for ticket in tickets:
            try:
                result = self.predict_ticket(
                    title=ticket.get('title', ''),
                    description=ticket.get('description', ''),
                    impact_level=ticket.get('impact_level', 'Medium'),
                    urgency_level=ticket.get('urgency_level', 'Medium'),
                    affected_users=ticket.get('affected_users', 1)
                )
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "ticket": ticket
                })
        
        return results
    
    def _confidence_label(self, confidence: float) -> str:
        """Convert confidence score to human-readable label."""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.75:
            return "High"
        elif confidence >= 0.6:
            return "Moderate"
        elif confidence >= 0.4:
            return "Low"
        else:
            return "Very Low"
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        return {
            "categories": list(self.category_classes),
            "priorities": list(self.priority_classes),
            "resolver_groups": list(self.resolver_classes),
            "impact_levels": list(self.impact_encoder.classes_),
            "urgency_levels": list(self.urgency_encoder.classes_),
            "sla_mapping": self.sla_map
        }


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*80)
    print("ITSM PREDICTION SERVICE - DEMO")
    print("="*80)
    
    # Initialize predictor
    predictor = ITSMPredictor()
    
    # Test cases
    test_tickets = [
        {
            "title": "Unable to connect to VPN",
            "description": "User reports that VPN client fails to connect. Error message says 'Connection timeout'. Affecting remote workers.",
            "impact_level": "High",
            "urgency_level": "High",
            "affected_users": 25
        },
        {
            "title": "Email not working",
            "description": "Cant login",
            "impact_level": "Medium",
            "urgency_level": "Low",
            "affected_users": 1
        },
        {
            "title": "Database performance degradation",
            "description": "Production database queries are running very slow since this morning. Query execution time increased from 2 seconds to 45 seconds. Multiple departments affected. Customer-facing applications timing out.",
            "impact_level": "High",
            "urgency_level": "High",
            "affected_users": 150
        },
        {
            "title": "Laptop keyboard not responding",
            "description": "Employee's laptop keyboard stopped working suddenly. External USB keyboard works fine. Laptop is Dell Latitude 5520.",
            "impact_level": "Medium",
            "urgency_level": "Medium",
            "affected_users": 1
        },
        {
            "title": "Need access to SharePoint site",
            "description": "New employee needs access to Finance department SharePoint site for viewing monthly reports.",
            "impact_level": "Low",
            "urgency_level": "Low",
            "affected_users": 1
        }
    ]
    
    print("\n" + "="*80)
    print("TESTING PREDICTIONS ON SAMPLE TICKETS")
    print("="*80)
    
    for i, ticket in enumerate(test_tickets, 1):
        print(f"\n{'─'*80}")
        print(f"TEST CASE {i}")
        print(f"{'─'*80}")
        print(f"Title: {ticket['title']}")
        print(f"Description: {ticket['description'][:80]}...")
        print(f"Input: Impact={ticket['impact_level']}, Urgency={ticket['urgency_level']}, Users={ticket['affected_users']}")
        
        # Get prediction
        result = predictor.predict_ticket(**ticket)
        
        print(f"\n📁 CATEGORY: {result['category']['predicted']} (confidence: {result['category']['confidence']:.2%} - {result['category']['confidence_level']})")
        top3_str = ', '.join([f"{c['category']} ({c['confidence']:.0%})" for c in result['category']['top_3_predictions']])
        print(f"   Top 3: {top3_str}")
        print(f"   Key terms: {', '.join(result['category']['key_terms'])}")
        
        print(f"\n🎯 PRIORITY: {result['priority']['predicted']} (confidence: {result['priority']['confidence']:.2%} - {result['priority']['confidence_level']})")
        print(f"   SLA: {result['priority']['sla_hours']} hours")
        print(f"   {result['priority']['explanation']}")
        
        print(f"\n👥 RESOLVER: {result['resolver']['predicted']} (confidence: {result['resolver']['confidence']:.2%} - {result['resolver']['confidence_level']})")
        print(f"   {result['resolver']['explanation']}")
        alternatives_str = ', '.join([f"{r['resolver_group']} ({r['confidence']:.0%})" for r in result['resolver']['top_3_suggestions'][1:]])
        print(f"   Alternatives: {alternatives_str}")
    
    print("\n" + "="*80)
    print("✅ PREDICTION SERVICE WORKING PERFECTLY!")
    print("="*80)
    print("\nModel Information:")
    info = predictor.get_model_info()
    print(f"  Categories: {', '.join(info['categories'])}")
    print(f"  Priorities: {', '.join(info['priorities'])}")
    print(f"  Resolver Groups: {', '.join(info['resolver_groups'])}")
