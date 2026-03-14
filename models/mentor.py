
import numpy as np


class MentorMixin:
    """
    Mixin compartido por todos los modelos sklearn.
    Proporciona _build_mentor_dataset() para construir un train set
    aumentado con réplicas de la instancia guiadas por el vector del mentor.
    """

    def _build_mentor_dataset(self, X_train, y_train,
                               instance, target_pred, mentor_vector,
                               n_replicas=15, base_weight=5.0,
                               noise_scale=0.05):
        """
        Parámetros
        ----------
        X_train, y_train : train set original
        instance         : instancia frontera a corregir
        target_pred      : clase correcta según la mayoría
        mentor_vector    : vector de importancia de features del mentor
        n_replicas       : réplicas de la instancia a añadir
        base_weight      : peso base de cada réplica vs ejemplos del train
        noise_scale      : escala del ruido gaussiano por réplica

        Devuelve X_aug, y_aug, w_aug listos para model.fit(..., sample_weight=w_aug)
        """
        x_inst = np.asarray(instance, dtype=float).reshape(1, -1)
        n_feat = x_inst.shape[1]

        mv = np.abs(mentor_vector[:n_feat]) if len(mentor_vector) >= n_feat \
             else np.abs(mentor_vector)
        mv_norm = mv / (mv.max() + 1e-8)

        # Peso proporcional a cuánto de discriminativas son las features del mentor
        importance_factor = 1.0 + float(mv_norm.mean()) * 3.0
        replica_weight    = base_weight * importance_factor

        replicas_x = [
            x_inst.flatten() + np.random.randn(n_feat) * noise_scale * mv_norm
            for _ in range(n_replicas)
        ]
        replicas_x = np.array(replicas_x, dtype=float)
        replicas_y = np.full(n_replicas, target_pred, dtype=y_train.dtype)
        replicas_w = np.full(n_replicas, replica_weight, dtype=float)

        X_aug = np.vstack([X_train, replicas_x])
        y_aug = np.concatenate([y_train, replicas_y])
        w_aug = np.concatenate([np.ones(len(X_train), dtype=float), replicas_w])

        return X_aug, y_aug, w_aug